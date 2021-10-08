import torch
import torch.nn as nn
import torch.nn.functional as F

''' gtsrb '''
num_classes = 43
in_channels = 3
input_size = 32

# ''' mnist '''
# num_classes = 10
# in_channels = 1
# input_size = 28

""" class model of target network for testing """
class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()

        self.num_classes=num_classes
        self.in_channels=in_channels
        self.input_size=input_size

        self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.linear = nn.Linear(32*int((self.input_size/4-3)*(self.input_size/4-3)), self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*int((self.input_size/4-3)*(self.input_size/4-3)))
        x = self.linear(x)
        return x

