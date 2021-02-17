CHANNELS = 3
MAPPING_WEIGHT = 2 # Multiplier for training mapping network in generator
LATENT_DIM = 512
IMG_SIZE = 1024

EPSILON = 1e-8 # Avoiding zero divisions
TRUNC_PSI = 0.5 # Psi for truncation
PPL_DECAY = 0.01 # For moving average of path lengths
R1REG_GAMMA = 10 # Weight of r1 loss term
PPL_BATCH = 2 # Reduce PPL batch size by this factor
PPL_WEIGHT = 2 # Weight for PPL loss

# Intervals (in terms of parameter updates)
D_R1REG_INTERVAL = 16 
G_PPL_INTERVAL = 4
ADA_INTERVAL = 256 

# Adaptive augmentation
ADA_INTERVAL = 256
AUG_P_TARGET = 0.6
ADA_LENGTH = 500 * 1000

# Training
USE_AMP = True
USE_AUGMENTS = True
BATCH_SIZE = 32
ITERATIONS = 20000
CHECKPOINT_INTERVAL = 50
SAMPLE_INTERVAL = 50
LOG_INTERVAL = 50


MIX_PROB = 0.9 # Mixing regularization
