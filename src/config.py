import torch

DATA_TEST_PATH = "./data/raw/test"
DATA_TRAIN_PATH = "./data/raw/train"
CHECKPOINT_DIR = "./checkpoints"

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 1
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
NUM_CHANNELS = 3
