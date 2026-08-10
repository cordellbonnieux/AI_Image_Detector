import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        loop = tqdm(loader, desc="Checking accuracy")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

            # live running accuracy in the bar itself
            running_acc = 100 * float(num_correct) / float(num_samples)
            loop.set_postfix(acc=f"{running_acc:.2f}%")

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")


def compute_confusion_matrix(loader, model, device, class_names=None, save_path="confusion_matrix.png"):
    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    num_classes = len(class_names)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        loop = tqdm(loader, desc="Computing confusion matrix")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, preds = outputs.max(1)

            for true_label, pred_label in zip(y.view(-1), preds.view(-1)):
                if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
                    cm[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.cpu().numpy(), interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max().item() / 2.0 if cm.max().item() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j].item() > thresh else "black"
            ax.text(j, i, f"{cm[i, j].item()}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Confusion matrix saved to {save_path}")
    return cm.cpu().numpy()