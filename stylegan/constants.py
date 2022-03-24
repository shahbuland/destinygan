# Architecture
CHANNELS = 3
LATENT_DIM = 512
IMG_SIZE = 512
N_MLP = 8 # Layers in mapping network
SWAGAN = True # Use SWAGAN models?

# Abandons stylegan entirely
# Makes constant input a random noise vector
SIMPLIFY_GEN = False

# Hyperparameters
LEARNING_RATE = 2e-3
BATCH_SIZE = 32
MICROBATCH_SIZE = 32 # Grad Accum is a must! (Ensure this divides BATCH_SIZE) 
EPSILON = 1e-8 # Avoiding zero divisions
MAPPING_WEIGHT = 2 # Map network weight during training
DEVICE = "cuda"

# Regularization and Truncation trick
MIX_PROB = 0.9 # Mixing regularization
PPL_DECAY = 1e-2
R1REG_GAMMA = 10 # Weight of R1 Loss term
PPL_WEIGHT = 2
D_R1REG_INTERVAL = 4
G_PPL_INTERVAL = 32
DO_R1_REG = True
DO_PPL_REG = False

TRUNC_PSI = -1 # -1 to disable trick
TRUNC_SAMPLES = 4096

# Debug
SKIP_MAPPING = False # Skip mapping network
RANDOM_INP = False # Random gen input instead of ConstantIn

# Training Flags
LOAD_CHECKPOINTS = True
USE_AUGMENTS = True # Differentiable augments before Disc.
ADAPT_AUG = False # Doesn't seem to do anything
DO_EMA = True
USE_WANDB = False
EMA_ALPHA = 0.005
LOAD_DATA = "ALL" # [ALL/SOME], [Put data on RAM/Use loader]
# Intervals and Iterations
ITERATIONS = 800000
CHECKPOINT_INTERVAL = 1000
SAMPLE_INTERVAL = 250
LOG_INTERVAL = 250

# Scaling for all augment probs
AUG_PROB = 0.297 # (1e-5 - len(dataset)) * 3e-6 (from lucidrains repo)
AUG_XFLIP = 1
AUG_ROT90 = 1
AUG_XINT = 1
AUG_SCALE = 1
AUG_ROT = 1
AUG_ANISO = 1
AUG_XFRAC = 1
AUG_BRIGHTNESS = 1
AUG_CONTRAST = 1
AUG_LUMAFLIP = 0 # Too extreme
AUG_HUE = 1
AUG_SAT = 1
