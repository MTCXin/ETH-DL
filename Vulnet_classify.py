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
from torch.utils.data import Dataset
from PIL import Image
import torch_dct as dct
from torchsampler import ImbalancedDatasetSampler
# from einops.layers.torch import Rearrange

#TODO BOOST

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CustomDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        with open(labels_file) as f:
            self.labels = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = list(self.labels.keys())

    def __len__(self):
        print('IMG Number:', len(self.img_paths))
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx].replace('../','')
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[self.img_paths[idx]]["l2_norm_class"]
        if self.transform:
            image = self.transform(image)
        if label not in [0,1,2]:
            print(label)
            raise ValueError
        return image, label
    

# class Sobel(nn.Module):
#     def __init__(self):
#         super(Sobel, self).__init__()
#         self.sobel_x = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float).view((1,1,3,3))
#         self.sobel_y = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float).view((1,1,3,3))
        
#         self.conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        
#         self.conv_x.weight = nn.Parameter(self.sobel_x)
#         self.conv_y.weight = nn.Parameter(self.sobel_y)
        
#         for param in self.conv_x.parameters():
#             param.requires_grad = False
#         for param in self.conv_y.parameters():
#             param.requires_grad = False
        
#         self.fc = nn.Sequential(nn.BatchNorm1d(224*224), 
#                                 nn.Linear(224*224, 512), 
#                                 nn.Dropout(0.2))
    
#     def forward(self, x):
#         edge_x = self.conv_x(x)
#         edge_y = self.conv_y(x)
#         edge = torch.sqrt(edge_x**2 + edge_y**2)
#         x = torch.flatten(edge, 1)
#         x = self.fc(x)
#         return x

 
# class ImageTransformer(nn.Module):
#     def __init__(self, image_size, patch_size, dim, num_heads, num_encoder_layers, num_decoder_layers):
#         super().__init__()
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = 3 * patch_size ** 2
#         self.patch_size = patch_size

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.transformer = nn.Transformer(d_model=dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
#         self.to_cls_token = nn.Identity()

#     def forward(self, img):
#         p = self.patch_size
#         x = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)(img)
#         x = self.patch_to_embedding(x)
#         x += self.pos_embedding
#         x = self.transformer(x, x)
#         x = self.to_cls_token(x[:, 0])
#         return x
 
      
# Model definition resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        
        # Tunnel_1: Resnet18
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = True
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

        # Tunnel_2: DCT
        # self.dct = nn.Sequential(
        #     nn.BatchNorm1d(224*224),
        #     nn.Linear(224*224, 512),
        #     nn.Dropout(0.2))
        
        # Tunnel_3: Sobel
        # self.Sobel = Sobel()
        
        # Tunnel_4: Transformer
        # self.transformer = ImageTransformer(
        #     image_size=224, patch_size=16, dim=512,
        #     num_heads=8, num_encoder_layers=6, num_decoder_layers=6
        # )
        
        
        # Full Connect
        mlp_input_size = resnet.fc.in_features
        self.fullconnect = nn.Sequential(
            nn.Linear(mlp_input_size, 3)# Changed to 4 for four classes
        )

    def forward(self, x):

        # Tunnel_1
        resnet_features = self.resnet_features(x)
        resnet_features = torch.flatten(resnet_features, 1)

        # Gray
        # grayscale = Grayscale(num_output_channels=1)
        # x_gray = grayscale(x)

        # Tunnel_2
        # dct_features = dct.dct_2d(x_gray)
        # dct_features = torch.flatten(dct_features, 1)
        # dct_features = self.dct(dct_features)
        
        # Tunnel_3
        # sobel_features = self.Sobel(x_gray)
        
        # Tunnel_4
        # Transformer_features = self.transformer(x)
        
        # Concatenate features
        combined_features = resnet_features
        x = self.fullconnect(combined_features)
        # x = F.softmax(x, dim=1)  # Apply softmax on the output
        return x


# Training Loop
def train_model(model, num_epochs, criterion, train_loader, test_loader):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            # pdb.set_trace()
            
            # pdb.set_trace()
            loss = criterion(outputs, labels)

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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            
            print(predicted)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    average_test_error = total_loss / len(test_loader.dataset)
    print(f'Average Test Error: {average_test_error:.4f}')
    print(f'Accuracy: {100 * correct_predictions / len(test_loader.dataset):.2f}%')
    return average_test_error


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CustomDataset(img_dir='./imgs/ImageNet1000', labels_file='./result_black_simba_class.json', transform=data_transforms)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

#TODO TRAINING PARAMS
learning_rate = 0.03
num_epochs = 30
batch_size = 256

model = CustomModel().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')  # Changed to CrossEntropyLoss

# criterion = nn.L1Loss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #, sampler=ImbalancedDatasetSampler(train_set))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

train_model(model, num_epochs, criterion, train_loader,test_loader)
test_model(model, test_loader)
