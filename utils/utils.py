import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DATA_DIR, IN_CHANNELS

def load_data(batch_size=64):
    
    #IN_CHANNELS is set to 1 because MNIST is black and white
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*IN_CHANNELS, std=[0.5]*IN_CHANNELS)
    ])

    mnist_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    mnist_test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=True)

    return dataloader, test_dataloader


def check_device():
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): # For Apple Silicon
        device = "mps"
    return device