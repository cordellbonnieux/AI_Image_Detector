import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ..models.cnn_model import AIDetectorCNN
from ..utils.accuracy import check_accuracy

"""
TODO - WIP!

This (training script) does not yet work the way we need it to.
Getting this to work is the first priority.

"""

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 784 #28x28 TODO change
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Define transforms (resize, normalize, etc.) TODO move to utils somewhere
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
])

# Load training data
train_dataset = datasets.ImageFolder(root="../../data/raw/train", transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load test data
test_dataset = datasets.ImageFolder(root="../../data/raw/test", transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init network
model = AIDetectorCNN(in_channels=1).to(device) #default num_classes=2; is AI or not

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the network
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

# check accuracy
check_accuracy(train_loader, model, device)
check_accuracy(test_loader, model, device)