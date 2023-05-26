import torch.nn as nn
import torch.nn.functional as F

# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    

# Testing CNN 2D
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4,stride = 4)
        self.batch = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size = 4, stride=1)
        self.fc1 = nn.Linear(in_features=40832, out_features=8)

    def forward(self, x):
        #x = x.unsqueeze(1) #We want only 1 channel as input
        print(x.shape)
        out = self.conv1(x)
        print(out.shape)
        out = self.batch(out)
        out = self.relu(out)
        print(out.shape)
        out = self.maxpool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        output = F.relu(out)

        return output


class CNNGH(nn.Module):
    def __init__(self):
        super(CNNGH, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=642,out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=8)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        return out.view(out.size(0), -1)


