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
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Grayscale
import numpy as np
import pdb
import torch.nn.init as init
# Check for GPU availability
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        with open(labels_file) as f:
            self.labels = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = list(self.labels.keys())

    def __len__(self):
        return len(self.img_paths)
        # return 2000


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[self.img_paths[idx]]["l2_norm"]

        if self.transform:
            image = self.transform(image)

        return image, label
    

# Custom DCT Layer
class DCTLayer(nn.Module):
    def __init__(self):
        super(DCTLayer, self).__init__()
        # Prepare indices for 7x7 DCT coefficients in a batched form
        self.indices = torch.tensor(np.indices((28, 28)).reshape(2, -1).T, dtype=torch.long)
        self.grayscale = Grayscale(num_output_channels=1)

    def forward(self, x):
        # Assuming x is a CPU tensor, move it to GPU if available
        x = x.to(device)

        # Convert to grayscale
        x_gray = self.grayscale(x)

        batch_size, _, height, width = x_gray.shape
        x_dct = torch.zeros(batch_size, 1, 28, 28, device=device)

        # Compute DCT
        for b in range(batch_size):
            x_channel = x_gray[b, 0, :, :]
            x_channel_dct = torch.fft.rfft2(x_channel)
            x_dct[b, 0, :, :] = x_channel_dct[self.indices[:, 0], self.indices[:, 1]].view(28, 28)

        return x_dct

       
# Model definition resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze the weights of ResNet18
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.dct_layer = DCTLayer()

        # Define the input size for the first layer of the MLP
        mlp_input_size = resnet.fc.in_features + 28 * 28

        # Three-layer MLP with Batch Normalization
        self.fc1 = nn.Linear(mlp_input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # BatchNorm for first layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # BatchNorm for second layer
        self.fc3 = nn.Linear(128, 1)    # Output layer

        # Initialize weights
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.to(device)

        # Extract features
        resnet_features = self.resnet_features(x)
        resnet_features = torch.flatten(resnet_features, 1)

        dct_features = self.dct_layer(x)
        dct_features = torch.flatten(dct_features, 1)

        # Concatenate features
        combined_features = torch.cat((resnet_features, dct_features), dim=1)

        # Pass through MLP with Batch Normalization
        x = F.relu(self.bn1(self.fc1(combined_features)))
        x = F.relu(self.bn2(self.fc2(x)))
        prediction = self.fc3(x)
        # pdb.set_trace()
        return prediction

# Define transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
dataset = CustomDataset(img_dir='./imgs/ImageNet1000', labels_file='./result_black_simba.json', transform=data_transforms)

# Split dataset
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Create the model
model = CustomModel()

# Training and Testing Loops (to be implemented)
# ...


# Hyperparameters
learning_rate = 1e-6
num_epochs = 1000
batch_size = 512

# Loss function and optimizer
criterion = nn.MSELoss()  # Since the task is to predict a positive float value
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Move model to GPU if available
model = CustomModel().to(device)

# Training Loop
def train_model(model, train_loader):
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

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Testing Loop
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            
            # Calculate the loss
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

    average_test_error = total_loss / total_count
    print(f'Average Test Error: {average_test_error:.4f}')
    return average_test_error

# Criterion for evaluation
criterion = nn.MSELoss()



# Run the training and testing loops
train_model(model, train_loader)
# Example of running the testing loop
average_test_error = test_model(model, test_loader)
print('\naverage_test_error:',average_test_error)

