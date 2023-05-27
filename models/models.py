import torch.nn as nn
import torch.nn.functional as F
import torch

# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        self.name="ConvNet"
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



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
        self.name="CNN"
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(in_features=1321984, out_features=8)
        self.fc2 = nn.Linear(in_features=1321984, out_features=8)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1) #We want only 1 channel as input
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        output = F.relu(out)
        return output
    

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.name="CNN2"
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2= nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(in_features=645, out_features=32)
        self.fc2 = nn.Linear(in_features=8192, out_features=8)

        self.dropout = nn.Dropout(p=0.3, inplace=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = x.unsqueeze(1) #We want only 1 channel as input
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.fc1(out)
        output = F.relu(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        output = F.relu(out)
        return output
    

class CNNGH(nn.Module):
        def __init__(self):
            super(CNNGH, self).__init__()
            self.name="CNNGH"
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16, kernel_size=(5),stride=2),
                nn.Dropout(0.5),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5),stride=2),
                nn.Dropout(0.5),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2)
            )
            self.fc1 = nn.Linear(in_features=642,out_features=64)
            self.fc2 = nn.Linear(in_features=64, out_features=8)

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


        def forward(self, x):
            x = x.unsqueeze(1)
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop(out)
            out = F.relu(out)
            out = self.fc2(out)
            out = F.relu(out)
            return out
        

class CNNGH1D(nn.Module):
        def __init__(self):
            super(CNNGH1D, self).__init__()
            self.name="CNNGH1D"
            self.layer1 = nn.Sequential(
                nn.Conv1d(in_channels=128,out_channels=128, kernel_size=4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer2 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer3 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2)
            )

            self.fc1 = nn.Linear(in_features=4864 ,out_features=8)
            self.dropout = nn.Dropout(0.3)
            
                
        def forward(self, x):
            #x = x.reshape(100,128,1291)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.dropout(out)
            return out
        
def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)