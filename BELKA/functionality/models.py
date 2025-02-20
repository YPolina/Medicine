import torch
import torch.nn as nn
import torch.optim as optim

class CNNBinaryClassifier(nn.Module):

    def __init__(self, input_dim = 142, hidden_dim = 32, kernel_size = 3, output_dim = 1):
        super(CNNBinaryClassifier, self).__init__()        

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride = 1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        
