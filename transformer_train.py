import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import torch.optim.lr_scheduler as lr_scheduler

import time
import os

from models.vqvae import VQVAE
from models.transformer import Transformer

from generator import generate_images
from utils.utils import load_data, check_device

from config import VQVAE_CONFIG, TRANSFORMER_CONFIG, TransformerTrainConfig
from config import TRANSFORMER_IMAGES_DIRECTORY, TRANSFORMER_WEIGHTS_PATH, VQVAE_WEIGHTS_PATH, SEED

#input: dataloader with images
#output: dataloader with tokenized images (images are encoded by vq vae encoder)
@torch.no_grad()
def images_to_tokens(vqvae, dataloader, batch_size=64, num_workers=0, device="cpu"):
    all_tokens = []
    all_class_labels = []
    offset = vqvae.num_embeddings
    vqvae.eval()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            tokens = vqvae.encode(images)  # [B, h, w, num_codebooks]
            tokens = tokens.cpu() # move to CPU to save memory
        tokens = tokens.view(tokens.size(0), -1) # [B, h * w * num_codebooks]
        all_tokens.append(tokens) 
        all_class_labels.append(labels)

    all_tokens = torch.cat(all_tokens, dim=0) #[num_images, h * w * num_codebooks ]
    all_labels = torch.cat(all_class_labels, dim=0).unsqueeze(-1) #[num_images, 1]
    
    #we insert the class label as the first token of the sequence
    #offset it by num_embeddings since vqvae tokens are in [0,...,num_embeddings-1]
    all_labels += offset
    combined = torch.cat((all_labels, all_tokens), dim=1) #[num_images, 1+  h * w * num_codebooks ]

    dataset = TensorDataset(combined)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader   # [Num images, 1+ h * w * num_codebooks]


def train_transformer_from_tokens(transformer, vqvae, optimizer, scheduler, dataloader, test_dataloader, device,
                                  epochs=1):
    
    #main training loop
    transformer.train()
    
    for epoch in range(epochs):
        total_loss = 0
        epoch_start_time = time.time()  # start epoch timer
        for batch_tokens, in dataloader:

            batch_tokens = batch_tokens.to(device).long()  # [B, seq_length]
            # Input tokens: all but last token 
            # Target tokens: all but first token 
            input_tokens = batch_tokens[:, :-1]    # [B, seq_length-1]  
            target_tokens = batch_tokens[:, 1:]    # [B, seq_length-1]

            # Forward pass
            optimizer.zero_grad()
            logits = transformer(input_tokens) # [B, seq_length-1, vocab_size]
            B, T, V = logits.shape
            logits_flat = logits.view(-1, V)
            targets_flat = target_tokens.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat) 

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
   
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / len(dataloader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        #print error on last training epoch
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.2f}, LR: {current_lr:.6f},  Time: {epoch_duration:.2f}s")

        #test and print error on validation set
        if (epoch+1) % 2 == 0:
            transformer.eval()
            total_loss = 0

            with torch.no_grad():
                for batch_tokens, in test_dataloader:
                    batch_tokens = batch_tokens.to(device).long()  # [B, seq_length]
                    
                    input_tokens = batch_tokens[:, :-1]    # [B, seq_length-1] 
                    target_tokens = batch_tokens[:, 1:]    # [B, seq_length-1]
                    logits = transformer(input_tokens)
                    
                    B, T, V = logits.shape
                    logits_flat = logits.view(-1, V)
                    targets_flat = target_tokens.reshape(-1)
        
                    loss = F.cross_entropy(logits_flat, targets_flat)
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(test_dataloader)
            print(f"Validation set loss: {avg_loss:.2f}")
            transformer.train()
                
        #draw 0-9 to see how its doing
        if (epoch+1) % 2 == 0:
            transformer.eval()
            draw_start_time = time.time() 
            results = generate_images(transformer, vqvae, num_images=10, temperature=0.3, device=device)
            draw_end_time = time.time() 
            print (f"Drew in {draw_end_time-draw_start_time} seconds")
            
            image_path = os.path.join(TRANSFORMER_IMAGES_DIRECTORY, f"transformer_image_{epoch+1}.png")
            save_image(results, image_path, normalize=True, scale_each=True, nrow=10)
            
            torch.save(transformer.state_dict(), TRANSFORMER_WEIGHTS_PATH)
            transformer.train()


    torch.save(transformer.state_dict(), TRANSFORMER_WEIGHTS_PATH)

def main():
    torch.manual_seed(SEED)
    device = torch.device(check_device())
    print(f"Using device: {device}")


    #define the models
    vqvae = VQVAE(**VQVAE_CONFIG).to(device)
    vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS_PATH, map_location=device))
    vqvae.eval()
    transformer = Transformer(**TRANSFORMER_CONFIG).to(device)
    transformer.train()

    #set up training configuration
    config = TransformerTrainConfig()
    #load the data 
    dataloader, test_dataloader = load_data()
    dataloader = images_to_tokens(vqvae, dataloader=dataloader, batch_size=config.batch_size, device=device)
    test_dataloader = images_to_tokens(vqvae, dataloader=test_dataloader, batch_size=config.batch_size, device=device)
    #set up training parameters
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    #train 
    train_transformer_from_tokens(transformer, vqvae, optimizer, scheduler, dataloader=dataloader, test_dataloader=test_dataloader,
                              epochs=config.num_epochs, device=device)

if __name__ == "__main__":
    main()