
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os

from models.vqvae import VQVAE  
from utils.utils import load_data, check_device

from config import VQVAE_CONFIG, VQVAE_WEIGHTS_PATH, VQVAE_IMAGES_DIRECTORY, SEED
from config import VQVAETrainConfig


#main training loop
def train(model, optimizer, dataloader, test_dataloader, device, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)
            x_recon, vq_loss = model(x)
            #train L1 loss on last few epochs: I found it helps make the reconstructions less blurry, less gray, and "sharper". 
            # I think it likes picking middle pixel values as big mistakes are punished heavily by L2 loss. 
            if epoch <= num_epochs-5:
                recon_loss = F.mse_loss(x_recon, x)
            else:
                recon_loss = F.l1_loss(x_recon, x) 
            loss = recon_loss + vq_loss 
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.2f}")
        # check loss on test set and view sample encodings 
        if (epoch + 1) % 2  == 0:
            model.eval()
            
            with torch.no_grad():
                for batch in test_dataloader:
                    x, _ = batch
                    x = x.to(device)

                    x_recon, vq_loss = model(x)
                    recon_loss = F.mse_loss(x_recon, x)
                    loss = recon_loss + vq_loss 
                    
            print(f"Validation loss: {loss.item():.2f}")   
            image_path = os.path.join(VQVAE_IMAGES_DIRECTORY, f"vqvae_image{epoch + 1}.png")
            save_image(torch.cat([x[:8], x_recon[:8]]), image_path, normalize=True, scale_each=True)
            model.train()
            torch.save(model.state_dict(), VQVAE_WEIGHTS_PATH)

    # Save model
    torch.save(model.state_dict(), VQVAE_WEIGHTS_PATH)

    # checking how many unique tokens over entire dataset
    all_tokens = []
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            tokens = model.encode(x)
        all_tokens.append(tokens.flatten())
    tokens = torch.cat(all_tokens)
    unique_tokens = tokens.unique()
    print(f"Used {len(unique_tokens)} / {model.num_embeddings} embeddings")

def main():
    torch.manual_seed(SEED)
    device = torch.device(check_device())
    #define the model
    model = VQVAE(**VQVAE_CONFIG).to(device)
    #get the training config
    config = VQVAETrainConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataloader, test_dataloader = load_data()
    print(f"Using device: {device}")
    train(model = model, optimizer=optimizer, dataloader= dataloader, test_dataloader=test_dataloader, device=device, num_epochs=config.num_epochs)

if __name__ == "__main__":
    main()