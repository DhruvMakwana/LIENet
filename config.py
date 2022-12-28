import torch

# paths
TRAIN_DIR = "../lol_dataset/our485"
VAL_DIR = "../lol_dataset/eval15"
TRAIN_LOGS_DIR = "logs/runs/train"
VAL_LOGS_DIR = "logs/runs/val"
RESULTS_DIR = "evaluations/"
CHECKPOINT_GEN_DIR = "checkpoints/best_gen.pth"
CHECKPOINT_CRITIC_DIR = "checkpoints/best_disc.pth"

# train hyperparameter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4
BATCH_SIZE = 4
NUM_WORKERS = 2
IMAGE_SIZE = (512, 512)
IN_CHANNELS = 3
NUM_EPOCHS = 500
CRITIC_ITERATIONS = 4
LAMBDA_GP = 10
LOAD_MODEL = False
SAVE_MODEL = True

# physical priors
CHANNEL_PRIOR_KERNEL = 15
TMIN=0.1
OMEGA=0.95
ALPHA=0.4
