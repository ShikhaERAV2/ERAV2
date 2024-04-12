import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from pytorch_grad_cam import GradCAM
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

# four plots are generated for train loss,train accuray ,test loss and test accuracy
def display_plot(train_losses,train_acc,test_losses,test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)

def get_transformation():
    train_transforms = transforms.Compose([
                                       transforms.RandomCrop(32,padding=4),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                           ])
    
    # Test Phase transformations
    test_transforms = transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
    return (train_transforms,test_transforms)

def get_misclassified_images(test_loader,classes,model,device):
    misclassified_images = []
    # Iterate through the test dataset and apply GradCAM
    for images, labels in iter(test_loader):
        #print(labels)
        # Run inference
        for img, label in zip(images,labels):
            outputs = model(img.to(device).unsqueeze(0))
            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            if classes[predicted] != classes[label]:
                #print("inside loop")
                misclassified_images.append((img,label,predicted))
                break
    return misclassified_images

def display_misclassified_images(misclassified_images,classes):
    # show only the first 10 images in 2,5 format
    fig, axes = plt.subplots(2, 5, figsize=(12,5))
    plt.subplots_adjust(bottom=2.3, top=2.7, hspace=1)
    i=0
    for img,label,predicted in misclassified_images:
        plt.subplot(2, 5, i + 1)
        plt.imshow(convert_to_imshow_format(img))
        plt.title('Target - ' + classes[label]+'\n'+'Predicted - '+ classes[predicted],fontsize = 8)
        i=i+1

def display_gradcam_images(model,misclassified_images,device,classes,target_layer):     
    
    # Denormalize the data using test mean and std deviation
    # Define the mean and standard deviation used for normalization
    mean = [0.5, 0.5, 0.5]  # Example mean values for RGB channels
    std = [0.5, 0.5, 0.5]   # Example standard deviation values for RGB channels
    denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    
    # Create the GradCAM instance
    cam = GradCAM(model=model, target_layers=target_layer)
    
    fig, axes = plt.subplots(2, 5, figsize=(12,5))
    plt.subplots_adjust(bottom=2.3, top=2.7, hspace=1)
    i=0
        
    
    # Iterate through the test dataset and apply GradCAM
    #for images, labels in iter(test_loader):
    for images, labels, predicted in misclassified_images[:10]:
        #print(classes[labels])
        #print(classes[predicted])
    
        images =  images.to(device).unsqueeze(0)
    
        # Generate CAM
        cam_image = cam(input_tensor=images, targets=None)
        grayscale_cam = cam(input_tensor=images, targets=None)
        
    
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
    
    
        # Get back the original image
        img = images.squeeze(0).to('cpu')
        img = denormalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()
    
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        #visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(visualization)
        plt.title('Target - ' + classes[labels]+'\n'+'Predicted - '+ classes[predicted],fontsize = 8)
        i=i+1


    
        
                