import torch

# paths
TRAIN_DIR = "E:/research/low-light/lol_dataset/our485"
VAL_DIR = "E:/research/low-light/lol_dataset/eval15"
TRAIN_LOGS_DIR = "logs/runs/train"
VAL_LOGS_DIR = "logs/runs/val"
RESULTS_DIR = "evaluations/"
CHECKPOINT_DIR = "checkpoints/best.pth"

# train hyperparameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
NUM_WORKERS = 2
IMAGE_SIZE = (256, 256)
IN_CHANNELS = 3
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True

# physical priors
CHANNEL_PRIOR_KERNEL = 15
TMIN=0.1
OMEGA=0.95