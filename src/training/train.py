import torch
import os
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from src.models.ai_detector import AIDetectorCNN
from src.utils.accuracy import check_accuracy, compute_confusion_matrix
from src.utils.preprocess import train_transform, test_transform
from src.config import CHECKPOINT_DIR, DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, DATA_TEST_PATH, DATA_TRAIN_PATH


# Load training data
train_dataset = datasets.ImageFolder(root=DATA_TRAIN_PATH, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load test data
test_dataset = datasets.ImageFolder(root=DATA_TEST_PATH, transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# init network with pretrained transfer learning
model = AIDetectorCNN(use_pretrained=True).to(DEVICE)

# define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# train the network
for epoch in range(EPOCHS):
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    model.train()
    epoch_loss = 0.0

    for batch_index, (data, targets) in enumerate(tqdm(train_loader, desc="Training")):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if len(train_loader) > 0:
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    else:
        print("Warning: training loader is empty; scheduler was not updated.")

# save model
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "ai_detector.pth"))
print(f"model saved to {CHECKPOINT_DIR}/ai_detector.pth")

# check accuracy
print("Checking accuracy on training data...")
check_accuracy(train_loader, model, DEVICE)
print("Checking accuracy on test data...")
check_accuracy(test_loader, model, DEVICE)
print("Computing confusion matrix on test data...")
compute_confusion_matrix(
    test_loader,
    model,
    DEVICE,
    class_names=["AI Generated", "Not AI Generated"],
    save_path="confusion_matrix.png",
)