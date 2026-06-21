import torch
import torch.nn as nn

class AIDetectorCNN(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 122 * 112, 1) # 0=real 1=AI
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 112 * 112)
        x = torch.sigmoid(self.fc1(x))
        return x
    