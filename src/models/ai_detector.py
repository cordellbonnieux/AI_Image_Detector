import torch.nn.functional as F
from torch import nn
from torchvision import models
from src.config import NUM_CLASSES, NUM_CHANNELS

class AIDetectorCNN(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, num_classes=NUM_CLASSES, use_pretrained=False):
        super(AIDetectorCNN, self).__init__()
        self.use_pretrained = use_pretrained

        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                self.model = models.resnet18(weights=weights)
            except Exception:
                self.model = models.resnet18(pretrained=True)

            if in_channels != 3:
                self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            return

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) #helps prevent overfitting

        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(num_classes)


    def forward(self, x):
        if self.use_pretrained:
            return self.model(x)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x