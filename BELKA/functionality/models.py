import torch
import torch.nn as nn
import torch.optim as optim

class CNNBinaryClassifier(nn.Module):

    def __init(self, input_dim = 142, hidden_dim = 32, kernel_size = 3, output_dim = 1, stride = 1, padding = 1):
        super(CNNBinaryClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride = stride, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=64, kernel_size=kernel_size, stride = stride, padding = padding)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool1d(kernel_size=2, stride = stride)
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
        self.fc2 = nn. Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x            





        
