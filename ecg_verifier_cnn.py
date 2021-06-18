'''
This class builds a CNN for ECG heartbeat verification
''' 
# Imports
import torch
from torch import nn
from copy import copy

class ECGVeriNet(nn.Module):

    def __init__(self, gpu = False):

        super(ECGVeriNet,self).__init__()

        # Declare GPU use
        self.gpu = gpu

        # Conv1
        self.conv1 = nn.Conv1d(in_channels  = 1,
                               out_channels = 96,
                               kernel_size = 49,
                               stride=1,
                               padding=0)

        # Pool
        self.pool = nn.MaxPool1d(kernel_size = 2,
                                  stride=2,
                                  padding=0)

        # Conv2
        self.conv2 = nn.Conv1d(in_channels  = 96,
                               out_channels = 128,
                               kernel_size = 25,
                               stride=1,
                               padding=0)

        # Conv3
        self.conv3 = nn.Conv1d(in_channels  = 128,
                               out_channels = 256,
                               kernel_size = 9,
                               stride=1,
                               padding=0)

        # Conv4
        self.conv4 = nn.Conv1d(in_channels  = 256,
                               out_channels = 512,
                               kernel_size = 9,
                               stride=1,
                               padding=0)

        # FC1
        self.fc1 = nn.Linear(3072, 4096)

        # FC2
        self.fc2 = nn.Linear(4096, 4096)

        # Dropout
        self.dropout = nn.Dropout()

        # FC 3
        self.fc3 = nn.Linear(4096, 1)

        # Output
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        
        # Feedforward
        #-------------------------------#
        # Conv Layers
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)

        x = torch.tanh(self.conv2(x))
        x = self.pool(x)

        x = torch.tanh(self.conv3(x))
        x = self.pool(x)

        x = torch.tanh(self.conv4(x))
        x = self.pool(x)

        # FC Layers
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        #-------------------------------#

        # Output
        return torch.sigmoid(x)


