import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from src.models.ai_detector import AIDetectorCNN
from src.utils.accuracy import check_accuracy
from src.utils.preprocess import image_transform
from src.config import CHECKPOINT_DIR, DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, DATA_TEST_PATH, DATA_TRAIN_PATH


# Load training data
train_dataset = datasets.ImageFolder(root=DATA_TRAIN_PATH, transform=image_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load test data
test_dataset = datasets.ImageFolder(root=DATA_TEST_PATH, transform=image_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# init network
model = AIDetectorCNN(in_channels=1).to(DEVICE)

# define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train the network
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

# save model
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "ai_detector.pth"))
print(f"model saved to {CHECKPOINT_DIR}/ai_detector.pth")

# check accuracy
check_accuracy(train_loader, model, DEVICE)
check_accuracy(test_loader, model, DEVICE)