from PIL import Image
import numpy as np
from src.config import IMAGE_SIZE
from torchvision import transforms

# NOTE unused
def preprocess_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Adjust to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For 3-channel images
])