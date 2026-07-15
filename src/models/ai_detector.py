import torch.nn.functional as F
from torch import nn
from src.config import NUM_CLASSES, NUM_CHANNELS

class AIDetectorCNN(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES):
        #in_channels: int: The number of channels in the input image. For MNIST, this is 1 (grayscale images).
        #num_classes: int: The number of classes we want to predict, in our case 2 AI or not AI.

        super(AIDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.LazyLinear(num_classes)

    def forward(self, x):
        #Parameters: x: torch.Tensor the input tensor.
        #Returns: torch.Tensor The output tensor after passing through the network.
        x = F.relu(self.conv1(x)) #conv + relu
        x = self.pool(x) #max pool
        x = F.relu(self.conv2(x))
        x = self.pool(x) 
        x = x.reshape(x.shape[0], -1) # Flatten the tensor
        x = self.fc1(x) # Apply fully connected layer
        return x