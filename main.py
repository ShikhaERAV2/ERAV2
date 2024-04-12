import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from models.resnet import *
from models.ModelTrainer import *
from utils import convert_to_imshow_format
#from model import ResNet_Custom


#!pip install torchsummary
from torchsummary import summary


def get_device():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def get_batch_size():
    BATCH_SIZE = 128
    return BATCH_SIZE

def get_epoch():
    EPOCHS = 20
    return EPOCHS

def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model,lr,weight_decay,betas,momentum):
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,betas = betas)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer

def get_correct_pred_count(prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

def train_model(model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_acc= []
    train_losses =[]
    
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        output = model(data)

        # Calculate loss
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()

        correct += get_correct_pred_count(output, target)
        processed += len(data)
        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return (train_losses,train_acc)

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_acc = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return(test_losses,test_acc)

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



