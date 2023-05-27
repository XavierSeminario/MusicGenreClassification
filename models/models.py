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
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer2 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer3 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.LeakyReLU(inplace=True),
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
                    
                    
 #LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(297888, 1291)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1291, 84)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.avgpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.avgpool2(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor

        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu4(out)

        out = self.fc3(out)
        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

<<<<<<< HEAD
class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels)
        )
    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.layer1(x) + self.layer2(x))


class ConvBlock2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock2d(1,16),
            nn.MaxPool2d(kernel_size=[8,2]),
            ResBlock2d(16,32),
            nn.MaxPool2d(kernel_size=[4,2]),
            ResBlock2d(32,64),
            nn.MaxPool2d(kernel_size=[4,2]),
            ResBlock2d(64,128)
        )

    def forward(self, x):
        return self.block(x)
    
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = ConvBlock2d()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128,8)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.reshape((x.shape[0], x.shape[3], x.shape[1]))
        out, _ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        return self.fc1(out)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)      
                    nn.init.zeros_(m.bias)

class CNN64(nn.Module):
    def __init__(self):
        super(CNN64, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(148016, 256)  # Reduce the number of neurons
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 64)  # Reduce the number of neurons
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor

        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fc3(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)      
                    nn.init.zeros_(m.bias)
