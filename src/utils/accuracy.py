import torch

def check_accuracy(loader, model, device):
    #if loader.dataset.train:
    #    print("Checking accuracy on training data")
    #else:
    #    print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        loop = tqdm(loader, desc="Checking accuracy")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

            #live running accuracy in the bar itself
            running_acc = 100 * float(num_correct) / float(num_samples)
            loop.set_postfix(acc=f"{running_acc:.2f}%")

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")