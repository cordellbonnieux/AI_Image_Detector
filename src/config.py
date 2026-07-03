import torch

DATA_TEST_PATH = "data/raw/test/"
DATA_TRAIN_PATH = "data/raw/train/"
CHECKPOINT_DIR = "checkpoints/"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
