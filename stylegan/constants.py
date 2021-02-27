CHANNELS = 3
MAPPING_WEIGHT = 2 # Multiplier for training mapping network in generator
LATENT_DIM = 512
IMG_SIZE = 256
N_MLP = 8 # Layers in mapping network
SWAGAN = True # Changes models to swagan models

LEARNING_RATE = 2e-3
EPSILON = 1e-8 # Avoiding zero divisions
TRUNC_PSI = 0.2 # Psi for truncation
TRUNC_SAMPLES = 4096 # Num samples to use for truncation mean calc
PPL_DECAY = 0.01 # For moving average of path lengths
R1REG_GAMMA = 10 # Weight of r1 loss term
PPL_BATCH = 2 # Reduce PPL batch size by this factor
PPL_WEIGHT = 2 # Weight for PPL loss

# Intervals (in terms of parameter updates)
D_R1REG_INTERVAL = 16 
G_PPL_INTERVAL = 4
ADA_INTERVAL = 256 

# Adaptive augmentation
AUG_P_TARGET = 0.6
ADA_LENGTH = 500 * 1000

# Training
USE_AMP = True
USE_AUGMENTS = False
BATCH_SIZE = 16
ITERATIONS = 200000
CHECKPOINT_INTERVAL = 1000
SAMPLE_INTERVAL = 500
LOG_INTERVAL = 100


MIX_PROB = 0.9 # Mixing regularization
