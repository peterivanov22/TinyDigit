import torch
import torch.nn.functional as F

from models.transformer import Transformer
from models.vqvae import VQVAE

from utils.utils import check_device
from torchvision.utils import save_image

from config import VQVAE_CONFIG, TRANSFORMER_CONFIG, TRANSFORMER_WEIGHTS_PATH, VQVAE_WEIGHTS_PATH
from config import TRANSFORMER_IMAGES_DIRECTORY
from config import LATENT_HEIGHT, LATENT_WIDTH,  SEQ_LENGTH, NUM_CODEBOOKS

@torch.no_grad()
def generate_images(transformer, vqvae, num_images=10, num_copies=1, temperature=0.5, device='cpu'):

    h, w = LATENT_HEIGHT, LATENT_WIDTH
    num_codebooks = NUM_CODEBOOKS
    seq_length = SEQ_LENGTH    
    offset = vqvae.num_embeddings
    all_images=[]

    for i in range(num_copies):
        # Initialize sampled token buffer
        sampled = torch.zeros(num_images, seq_length, dtype=torch.long, device=device)

        class_labels = torch.arange(10, dtype=torch.long, device=device) 
        class_labels += offset
        sampled[:, 0] = class_labels  # insert class token at position 0


        # Initialize batch of sampled sequences: [num_images, 1]
        #sampled = torch.randint(0, vocab_size, (num_images, 1), device=device)

        for i in range(1, seq_length):
            logits = transformer(sampled[:, :i])  # feed partial sequence
            next_token_logits = logits[:, -1, :]  # logits for next token
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sampled[:, i] = next_token.squeeze(-1)

        # Reshape to latent grid
        sampled = sampled[:,1:] #remove class token
        sampled = sampled.view(num_images, h, w, num_codebooks)
        images = vqvae.decode(sampled)  # [num_images, C, H, W]
        all_images.append(images)
    
    all_images = torch.cat(all_images, dim=0)  # shape: [100, C, H, W]
    return all_images.cpu()


def main():
    device = torch.device(check_device())
    torch.manual_seed(42)
    #define the model
    vqvae = VQVAE(**VQVAE_CONFIG).to(device)
    vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS_PATH, map_location=device))
    vqvae.eval()

    transformer = Transformer(**TRANSFORMER_CONFIG).to(device)    
    transformer.load_state_dict(torch.load(TRANSFORMER_WEIGHTS_PATH, map_location=device))
    transformer.eval()

    #can play with the temperature
    temperature = 0.5
    results = generate_images(transformer, vqvae, num_images=10, num_copies=10, temperature=temperature, device=device)
    #image_path = os.path.join(TRANSFORMER_IMAGES_DIRECTORY, "generated.png")
    save_image(results, f"generated_temp_{temperature:.2f}.png", normalize=True, scale_each=True, nrow=10)

if __name__ == "__main__":
    main()