from torch.utils.data import DataLoader
from torchvision import datasets
import torch, os
from src.models.ai_detector import AIDetectorCNN
from src.utils.preprocess import test_transform
from src.utils.accuracy import check_accuracy, compute_confusion_matrix
from src.config import CHECKPOINT_DIR, DEVICE, DATA_TEST_PATH, BATCH_SIZE

model = AIDetectorCNN(use_pretrained=True).to(DEVICE)
ckpt = os.path.join(CHECKPOINT_DIR, "ai_detector.pth")
state = torch.load(ckpt, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

test_dataset = datasets.ImageFolder(root=DATA_TEST_PATH, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Checking accuracy on test set...")
check_accuracy(test_loader, model, DEVICE)
print("Saving confusion matrix to confusion_matrix.png...")
compute_confusion_matrix(test_loader, model, DEVICE,
                         class_names=["AI Generated", "Not AI Generated"],
                         save_path="confusion_matrix.png")