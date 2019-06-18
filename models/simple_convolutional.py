import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_Convolutional(nn.Module):
    def __init__(self, input_shape):
        self.new_input_shape = input_shape
        
        super(Simple_Convolutional, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, kernel_size=5, stride=1)        
        self.maxpool1 = nn.MaxPool1d(kernel_size=5)
        self.new_input_shape = (input_shape - 4) // 5
        
        self.conv2 = nn.Conv1d(20, 50, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        self.new_input_shape = (self.new_input_shape - 4) // 3
        
        self.conv3 = nn.Conv1d(50, 100, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)
        self.new_input_shape = (self.new_input_shape - 2) // 3
        
        self.fc1 = nn.Linear(100 * self.new_input_shape, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        
        x = x.view(-1, 100 * self.new_input_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x