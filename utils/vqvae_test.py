import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.vqvae import VQVAE

from utils.utils import check_device
from config import VQVAE_CONFIG, VQVAE_WEIGHTS_PATH, DATA_DIR, SEED

def load_image():

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*1, std=[0.5]*1)
        ])
    mnist_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    image, _ = mnist_dataset[0] #image is [C,H,W]
    return image

def main():
    torch.manual_seed(SEED)
    device = torch.device(check_device())
    print(f"Using device: {device}")

    #define the model
    vqvae = VQVAE(**VQVAE_CONFIG).to(device)
    vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS_PATH, map_location=device))
    vqvae.eval()
    
    images =[]
    image = load_image()
    image = image.unsqueeze(0)
    image= image.to(device)

    tokens = vqvae.encode(image)  # [1, h, w, num_codebooks]
    print(torch.unique(tokens).numel())
    decoded_image = vqvae.decode(tokens)  # [1, C, H, W]
    images.append(image.cpu())
    images.append(decoded_image.cpu())

    images = torch.cat(images, dim=0)
    save_image(images, "./utils/vqvae_test.png", normalize=True, scale_each=True)

if __name__ == "__main__":
    main()