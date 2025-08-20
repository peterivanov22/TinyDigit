from dataclasses import dataclass
from pathlib import Path


SEED = 42


# ====== VQVAE Parameter ======

#I found that NUM_EMBEDDINGS = 64 is essentially the same, but seemed to hurt the transformer a little
#32 starts to hurt the VQVAE reconstructions
NUM_EMBEDDINGS = 128
EMBEDDING_DIM = 64
#Can increase this for more power in the encoder
HIDDEN_DIM = 64
#Number of residual blocks to use in encoder and decoder
NUM_RES_BLOCKS = 2
#If RBG change this to 3 (MNIST is BLACK and WHITE)
IN_CHANNELS = 1

# a [3, H, W] image is converted to a [l_h, l_w, num_codebooks] latent by the VAE encoder
# MNIST is 28x28, and we downsample to 4x4x1. 
# Then the transformer takes in 1+16 = 17 tokens (16 from the latent grid and 1 token at the start denoting the class label)
# Each token is an integer in [0, NUM_EMBEDDINGS-1] and the class label tokens are in [NUM_EMBEDDINGS, NUM_EMBEDDING-NUM_CLASSES-1]

LATENT_HEIGHT = 4
LATENT_WIDTH = 4
NUM_CODEBOOKS = 1


# ====== VQ-VAE Config ======
VQVAE_CONFIG = {
    "in_channels": IN_CHANNELS,
    "num_codebooks": NUM_CODEBOOKS,
    "num_embeddings": NUM_EMBEDDINGS,
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_res_blocks": NUM_RES_BLOCKS
}

# ====== Transformer Parameters ======
SEQ_LENGTH = LATENT_HEIGHT*LATENT_WIDTH*NUM_CODEBOOKS + 1 # +1 is for first token which represents class
#This scales the training time linearly
D_MODEL = 256
#Increasing this seems to be slightly beneficial
N_HEADS = 4
N_LAYERS = 4
#10 Digits in MNIST
NUM_CLASSES = 10  

TRANSFORMER_CONFIG = {
    "vocab_size": NUM_EMBEDDINGS + NUM_CLASSES,
    "seq_length": SEQ_LENGTH,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "latent_height": LATENT_HEIGHT,
    "latent_width": LATENT_WIDTH,
    "num_classes": NUM_CLASSES,
    "num_codebooks": NUM_CODEBOOKS
}


# ====== Training Parameters ======

class VQVAETrainConfig:
    learning_rate: float = 2e-4
    batch_size: int = 64
    num_epochs: int = 30

class TransformerTrainConfig:
    lr: float = 2e-3
    batch_size: int = 128
    #Epochs can be decreased further, it pretty quickly generates decent looking digits
    num_epochs: int = 40
    step_size: int = 5
    gamma: float = 0.75
    weight_decay: float = 1e-5
    num_workers: int = 0
    epoch_info_interval: int = 1
    test_set_interval: int = 2
    draw_interval: int  = 2
        

# ====== Base Paths ======
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "_data"
WEIGHTS_DIR = BASE_DIR / "_weights"
RESULTS_DIR = BASE_DIR / "_results"

# ====== Specific File Paths ======
VQVAE_WEIGHTS_DIRECTORY = WEIGHTS_DIR / "vqvae"
VQVAE_WEIGHTS_PATH = VQVAE_WEIGHTS_DIRECTORY / "vqvae_weight.pth"

VQVAE_IMAGES_DIRECTORY = RESULTS_DIR / "vqvae"

TRANSFORMER_WEIGHTS_DIRECTORY = WEIGHTS_DIR / "transformer"
TRANSFORMER_WEIGHTS_PATH = TRANSFORMER_WEIGHTS_DIRECTORY / "transformer_weight.pth"

TRANSFORMER_IMAGES_DIRECTORY = RESULTS_DIR / "transformer"
