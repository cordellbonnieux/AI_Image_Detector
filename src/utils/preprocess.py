import os
import cv2
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(244, 244)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array