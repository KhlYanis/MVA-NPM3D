
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
        
class RandomResize(object):
    def __call__(self, pointcloud):
        size_factor = random.random()*1.5 + 0.5   # factor between 1/2 and 2
        resized_pointcloud = pointcloud * size_factor
        return resized_pointcloud
        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    #return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])
    return transforms.Compose([RandomRotation_z(),RandomNoise(),RandomResize(),ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE
        self.classes = classes

        # Initialize the layers
        ## LAYER 1
        self.layer1 = nn.Sequential(
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        ## LAYER 2
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout1d(p = 0.3),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        ## Classification layer
        self.final_layer = nn.Linear(256, self.classes)

    def forward(self, input):
        # YOUR CODE
        x = input.flatten(1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final_layer(x)

        return x





class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE

        self.classes = classes

        ## BLOCK 1 :
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        ## BLOCK 2 :
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels = 128, out_channels = 1024, kernel_size = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Max Pooling layer
        self.pooling = nn.MaxPool1d(kernel_size= 1024)

        ## BLOCK 3 :
        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p = 0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.classes)
        )
    
    def forward(self, input):
        # YOUR CODE

        x = self.block1(input)
        x = self.block2(x)
        x = self.pooling(x)
        x = self.block3(torch.squeeze(x))

        return x
        
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # YOUR CODE

        ## BLOCK 1 :
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels = 128, out_channels = 1024, kernel_size = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Pooling Layer
        self.pooling = nn.MaxPool1d(1024)

        ## BLOCK 2 : 
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )

    def forward(self, input):
        # YOUR CODE
        x = self.block1(input)
        x = self.pooling(x)
        x = self.block2(torch.squeeze(x))
        x = x.view(-1, self.k, self.k)
        return x

class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE
        self.classes = classes

        self.tnet1 = Tnet(k=3)

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        ## BLOCK 2 :
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels = 128, out_channels = 1024, kernel_size = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Max Pooling layer
        self.pooling = nn.MaxPool1d(kernel_size= 1024)

        ## BLOCK 3 :
        self.block3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p = 0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.classes)
        )


    def forward(self, input):
        # YOUR CODE
        m3x3 = self.tnet1(input)
        x = torch.bmm(torch.transpose(input,1,2), m3x3).transpose(1,2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pooling(x)
        x = self.block3(torch.squeeze(x))

        return x, m3x3


def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            #outputs = model(inputs.transpose(1,2))
            outputs, m3x3 = model(inputs.transpose(1,2))
            #loss = basic_loss(outputs, labels)
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    #outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "TP6/data/ModelNet10_PLY/ModelNet10_PLY/"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = MLP()
    #model = PointNetBasic()
    model = PointNetFull()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device);
    
    train(model, device, train_loader, test_loader, epochs = 10)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


