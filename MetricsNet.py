import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import torchvision.datasets as datasets
import json
import pandas as pd
import os
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torchvision.transforms import Grayscale
import numpy as np
import pdb
import torch.nn.init as init
from torch.utils.data import Dataset
from PIL import Image
import torch_dct as dct
from torchsampler import ImbalancedDatasetSampler

#TODO BOOST

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CustomDataset(Dataset):
    def __init__(self, data, labels):

        self.data = pd.read_csv(data, index_col=0)
        self.data = self.data.drop(["Local_Binary_Patterns_4", "Local_Binary_Patterns_5", "Gabor_Filters_8"], axis=1)
        self.data = (self.data - self.data.mean()) / self.data.std()
        
        self.data = self.data.values
        
        
        self.labels = pd.read_csv(labels, index_col=0).values[:, 0]

        
    def __len__(self):
        return len(self.labels)
        # return 2000


    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if label == 0:
            label = 15
            print('CHANGED')
        
        sample = torch.from_numpy(sample).to(torch.float)
        # label = torch.tensor(label, dtype=torch.float)
        
        if 0 < label <= 5:
            label = 0
        elif 5 < label <= 10:
            label = 1
        else:
            label = 2
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label

# Model definition resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.fullconnect = nn.Sequential(
            nn.Linear(155, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        
        x = self.fullconnect(x)
        return x


# Training Loop
def train_model(model, num_epochs, criterion, train_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            
            # Forward pass
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            
            # loss = criterion(outputs, labels.view(-1, 1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)

        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        print(f'Accuracy: {100 * correct_predictions / len(train_loader.dataset):.2f}%')
        test_model(model, test_loader)
        
# Testing Loop
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            
            # print(outputs)
            # loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()

    average_test_error = total_loss / len(test_loader.dataset)
    print(f'Average Test Error: {average_test_error:.4f}')
    print(f'Accuracy: {100 * correct_predictions / len(test_loader.dataset):.2f}%')
    return average_test_error


# data_transforms = transforms.Compose([
#     transforms.ToTensor()
# ])
dataset = CustomDataset(data='./X.csv', labels='./Y.csv')
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

#TODO TRAINING PARAMS
learning_rate = 0.003
num_epochs = 50
batch_size = 256

model = CustomModel().to(device)
# criterion = nn.MSELoss(reduction='sum') 
# criterion = nn.L1Loss(reduction='sum')
criterion = nn.CrossEntropyLoss(reduction='sum') 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #, sampler = ImbalancedDatasetSampler(train_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

train_model(model, num_epochs, criterion, train_loader)
test_model(model, test_loader)
