CHANNELS = 3
MAPPING_WEIGHT = 2 # Multiplier for training mapping network in generator
LATENT_DIM = 512
IMG_SIZE = 1024

EPSILON = 1e-8 # Avoiding zero divisions
TRUNC_PSI = 0.5 # Psi for truncation
PPL_DECAY = 0.01 # For moving average of path lengths
R1REG_GAMMA = 2
AUG_P = 2 # Augment probability

# Intervals (in terms of parameter updates)
D_R1REG_INTERVAL = 2 
G_PPL_INTERVAL = 2

# Training
USE_CUDA = True
USE_HALF = True
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 50
SAMPLE_INTERVAL = 50
LOG_INTERVAL = 50
