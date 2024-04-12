from tqdm import tqdm
from torchvision import datasets, transforms
from models.resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

#from models.ModelTrainer import *

class ModelTrainer:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def train(self, model, device, train_loader, optimizer, epoch,criterion):
        model.train()
        pbar = tqdm(train_loader)

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

            correct += self.get_correct_pred_count(output, target)
            processed += len(data)
            pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        self.train_acc.append(100*correct/processed)
        self.train_losses.append(train_loss/len(train_loader))

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def display_plot(self):
      fig, axs = plt.subplots(2,2,figsize=(15,10))
      axs[0, 0].plot(self.train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1, 0].plot(self.train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(self.test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1, 1].plot(self.test_acc)
      axs[1, 1].set_title("Test Accuracy")

       
