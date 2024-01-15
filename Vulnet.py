import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import torchvision.datasets as datasets
import json
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
    def __init__(self, img_dir, labels_file, transform=None):
        # super(CustomDataset, self).__init__()
        
        with open(labels_file) as f:
            self.labels = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = list(self.labels.keys())

    def __len__(self):
        return len(self.img_paths)
        # return 2000


    def __getitem__(self, idx):
        img_path = self.img_paths[idx].replace('../','')
        image = Image.open(img_path).convert('RGB')
        label = self.labels[self.img_paths[idx]]["l2_norm"]
        if label == 0:
            label = 15
        if self.transform:
            image = self.transform(image)
    
        return image, label

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float).view((1,1,3,3))
        self.sobel_y = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float).view((1,1,3,3))
        
        self.conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.conv_x.weight = nn.Parameter(self.sobel_x)
        self.conv_y.weight = nn.Parameter(self.sobel_y)
        
        for param in self.conv_x.parameters():
            param.requires_grad = False
        for param in self.conv_y.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(nn.BatchNorm1d(224*224), 
                                nn.Linear(224*224, 512), 
                                nn.Dropout(0.2))
    
    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        x = torch.flatten(edge, 1)
        x = self.fc(x)
        return x
      
# Model definition resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        
        # Tunnel_1: Resnet18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

        # Tunnel_2: DCT
        self.dct = nn.Sequential(
            nn.Linear(224*224, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU())
        
        # Tunnel_3: Sobel
        # self.Sobel = Sobel()
        
        # Full Connect
        mlp_input_size = 512 + 512
        self.fullconnect = nn.Sequential(
            nn.Linear(mlp_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  
            nn.Linear(128, 1),
            nn.ReLU()
        )


        # # Initialize weights
        # init.kaiming_uniform_(self.fullconnect[0].weight, nonlinearity='relu')
        # init.kaiming_uniform_(self.fullconnect[2].weight, nonlinearity='relu')
        # init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        # init.xavier_uniform_(self.fc3.weight)
        # init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):

        # Tunnel_1
        resnet_features = self.resnet_features(x)
        resnet_features = torch.flatten(resnet_features, 1)

        # Gray
        grayscale = Grayscale(num_output_channels=1)
        x_gray = grayscale(x)
        # x_gray = torch.squeeze(x_gray)

        # Tunnel_2
        dct_features = dct.dct_2d(x_gray)
        dct_features = torch.flatten(dct_features, 1)
        dct_features = self.dct(dct_features)
        
        # Tunnel_3
        # sobel_features = self.Sobel(x_gray)
        
        # Concatenate features
        combined_features = torch.cat((resnet_features,dct_features), dim=1)
        x = self.fullconnect(combined_features)
        return x


# Training Loop
def train_model(model, num_epochs, criterion, train_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            # Forward pass
            outputs = model(images)

            
            loss = criterion(outputs, labels.view(-1, 1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        test_model(model, test_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Testing Loop
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            
            # print(outputs)
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()

    average_test_error = total_loss / len(test_loader.dataset)
    print(f'Average Test Error: {average_test_error:.4f}')
    return average_test_error


data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CustomDataset(img_dir='./imgs/ImageNet1000', labels_file='./result_black_simbav2.json', transform=data_transforms)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

#TODO TRAINING PARAMS
learning_rate = 0.0003
num_epochs = 50
batch_size = 256

model = CustomModel().to(device)
# criterion = nn.MSELoss(reduction='sum') 
criterion = nn.L1Loss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #, sampler = ImbalancedDatasetSampler(train_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

train_model(model, num_epochs, criterion, train_loader)
test_model(model, test_loader)
