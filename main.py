from PIL import Image
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Subset

import torch
import torch.optim as optim

import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from model import GPTArtModel

from utils import create_training_data_from_dataset
from tqdm.auto import tqdm as auto_tqdm 



# hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2
image_size = 224


#tuna 
LEARNING_RATE = 0.001           # Learning rate for the AdamW optimizer.
EPOCHS = 3                     # Number of training epochs for each model.
BATCH_SIZE_TRAIN = 512 
# ------------

_device_type = "cpu" # Default
if torch.cuda.is_available():
    _device_type = "cuda"
    print("Global Config: CUDA (GPU) is available. Using CUDA.")
elif torch.backends.mps.is_available(): # For Apple Silicon
    _device_type = "mps"
    print("Global Config: MPS (Apple Silicon GPU) is available. Using MPS.")
else:
    print("Global Config: No GPU detected. Using CPU.")
device = torch.device(_device_type) # Define the global device variable



torch.manual_seed(1337)

#------------LOAD DATA HERE
train_path = './images/train/Monet/'
test_path = './images/test/Monet/'



example_image = train_path + '211709.jpg'
image = Image.open(example_image)




#print(img_flat.shape)


#load data  -------------------

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.PILToTensor(), # Converts to [0,255] int tensor
    transforms.Lambda(lambda x: x.long())   # shape: (C, H, W), dtype: torch.long
])

train_monet_path = "./images/train_monet"

train_ds = datasets.ImageFolder(train_monet_path, transform=transform)

tensor, label = train_ds[0]
print(tensor.shape)
#each element is # shape: (C, H, W), dtype: torch.long

train_contexts, train_targets = create_training_data_from_dataset(train_ds, context_length=block_size)


print( "\nModel V1 - Data Shapes:")
print(f"  train_contexts shape: {train_contexts.shape}, dtype: {train_contexts.dtype}")
print(f"  train_targets shape: {train_targets.shape}, dtype: {train_targets.dtype}")

train_contexts = train_contexts.long()
train_targets = train_targets.long()

train_contexts = train_contexts.to(device)
train_targets = train_targets.to(device)

#initialize and train model below - 
model = GPTArtModel(n_embd=n_embd, block_size=block_size)
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


""" @torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            losses[k]=loss.item()
        out[split] = losses.mean()
    model.train()
    return out """


#------------------------------
#tiny shaklespeare trainn below::
""" for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch()

    #evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
 """

#------------------------

# --- Model V1, Data, Loss, Optimizer ---


criterion = nn.CrossEntropyLoss()
nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

n_samples = len(train_contexts)
if n_samples == 0:
    print("No training samples available for Model V1. Skipping training.")
else:
    print(f"\nTraining Model V1 (OneHotPixelPredictor) on {n_samples:,} samples for {EPOCHS} epochs.")
    print(f"Predicting 1 RBG value.")

    # --- Training Loop for Model V1 ---
    epoch_pbar = auto_tqdm(range(EPOCHS), desc="Model V1 Training Epochs", position=0, leave=True)
    for epoch in epoch_pbar:
        model.train() # Set model to training mode
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle indices for each epoch for batching from the large tensor
        indices = torch.randperm(n_samples, device=device)
        
        # Calculate total number of batches for the inner progress bar
        total_batches_in_epoch = (n_samples + BATCH_SIZE_TRAIN - 1) // BATCH_SIZE_TRAIN
        batch_pbar = auto_tqdm(range(0, n_samples, BATCH_SIZE_TRAIN), 
                             desc=f"Epoch {epoch+1}/{EPOCHS}", 
                             position=1, leave=False, 
                             total=total_batches_in_epoch)

        for start_idx in batch_pbar:
            end_idx = min(start_idx + BATCH_SIZE_TRAIN, n_samples)
            if start_idx == end_idx: continue # Skip if batch is empty

            batch_indices = indices[start_idx:end_idx]
            
            batch_context_tokens = train_contexts[batch_indices]  # Integer tokens
            batch_target_tokens = train_targets[batch_indices]    # Integer tokens (3 RBG VALUES)
            

            optimizer.zero_grad()
            
            # Model V1 forward pass - x_tokens are integer tokens, training=True
            logits = model(batch_context_tokens, training=True) 
            B,T,E = logits.shape
            logits = logits.view(B*T, E)
            split_logits = torch.split(logits,[256,256,256], dim=-1 )
            r,g,b = split_logits #(B*T, 256)
            B, T, C = batch_target_tokens.shape 
            batch_target_tokens = batch_target_tokens.reshape(B*T, C)
          
            loss_r = criterion(r, batch_target_tokens[:,0])
            loss_g = criterion(g, batch_target_tokens[:,1])
            loss_b = criterion(b, batch_target_tokens[:,2])
            loss = loss_r +loss_g + loss_b
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0: # Update progress bar postfix less frequently
                 batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if num_batches > 0: # Avoid division by zero if n_samples was small
            avg_loss = epoch_loss / num_batches
            epoch_pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        else:
            epoch_pbar.set_postfix(avg_loss="N/A")


#--------------------------------


def sample():
    pass



#context = torch.zeros((3,), dtype = torch.long)
#context = context.unsqueeze(0).unsqueeze(0)
#result = m.generate(context, max_new_tokens=224*20-1)

#result = result[0].permute(1,0)
#C, P = result.shape
#result = result.reshape(C, P//224, 224)

#image_result = tensor_to_pil(result)
#display(image_result)