import torch
import torch.nn as nn
import torch.nn.functional as F

# input (B, hidden_dim, H, W)
# output (B, hidden_dim, H, W)
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
    def forward(self, x):
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
       
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

    def forward(self, z):  # z: (B, embedding_dim, H, W)
        z_flat = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, embedding_dim]
        z_flat = z_flat.view(-1, self.embedding_dim)  # [B*H*W, embedding_dim]

        # Compute distances
        distances = (z_flat ** 2).sum(dim=1, keepdim=True) \
                    - 2 * z_flat @ self.embedding.weight.t() \
                    + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True) #[B*H*W, num_embeddings]
 
        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1) #[BHW]
        quantized = self.embedding(encoding_indices).view(z.shape) # [B, H, W, embedding_dim]

        encoding_loss = torch.mean((z.detach() - quantized) ** 2)
        commitment_loss = torch.mean((z - quantized.detach()) ** 2)
        loss = encoding_loss + self.commitment_cost *commitment_loss
       
        #Can try this simpler loss, works reasonably well
        #loss =  F.mse_loss(quantized, z)

        quantized = z + (quantized - z).detach()  # straight-through estimator
        return quantized, loss, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]) # encoding_indices: [B, H, W]

# Input: (B, in_channels, 28, 28) 
# Output (B, latent_dim, 4, 4)
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim , num_codebooks, num_res_blocks):
        super().__init__()
        self.latent_dim = embedding_dim * num_codebooks
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1), #(B, hidden_dim, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), #(B, hidden_dim, H/4, W/4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), #(B, hidden_dim, 4, 4),
            nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_blocks)]),  # (B, hidden_dim, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.latent_dim, 1)                          # (B, latent_dim, H/4, W/4)

        )
    def forward(self, x): 
        x = self.encode(x)
        return x
    
# Input (B, latent_dim, 4, 4)
# Input: (B, in_channels, 28, 28) 
class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dim, embedding_dim, num_codebooks, num_res_blocks):
        super().__init__()
        self.latent_dim = embedding_dim * num_codebooks
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, hidden_dim, 1),                          # (B, latent_dim, 4, 4)
            nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_blocks)]), #(B, latent_dim, 4, 4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1), #(B, latent_dim, 7, 7) 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1), #(B, latent_dim, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1), # (B, out_channels, H, W)
        )
    def forward(self, x):  
        x = self.decode(x) 
        x = torch.tanh(x) 
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels, num_codebooks, num_embeddings, embedding_dim, hidden_dim, num_res_blocks ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_embeddings = num_embeddings
        self.encoder = Encoder(in_channels=in_channels, num_codebooks=num_codebooks, hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_res_blocks=num_res_blocks)
        self.vq_layers = nn.ModuleList([VectorQuantizer(num_embeddings, embedding_dim) for _ in range(num_codebooks)])
        self.decoder = Decoder(out_channels=in_channels, num_codebooks=num_codebooks, hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_res_blocks=num_res_blocks)

    def forward(self, x):  # x: (B, in_channels, H, W) 
        z_e = self.encoder(x)
        z_chunks = torch.chunk(z_e, self.num_codebooks, dim=1) #list of num_codebooks tensors, each (B,embedding_dim, H/4, W/4)
        
        quantized_chunks = []
        encoding_indices_list = []
        vq_loss_list = []

        for i, vq in enumerate(self.vq_layers):
            z_q, vq_loss, encoding_indices = vq(z_chunks[i])
            vq_loss_list.append(vq_loss)
            quantized_chunks.append(z_q)
            encoding_indices_list.append(encoding_indices)
       
        quantized = torch.cat(quantized_chunks, dim=1)  # Concatenate along the last dimension
        x_recon = self.decoder(quantized)
        vq_loss = sum(vq_loss_list) / len(vq_loss_list)
        return x_recon, vq_loss

    #image to tokens
    # x: (B, in_channels, H, W) 
    # output  [B, 4, 4, num_codebooks]. Each entry is token (aka integer in [1,...,num_embeddings])
    def encode(self, x): 
        z_e = self.encoder(x) # (B, latent_dim, 4, 4)
        z_chunks = torch.chunk(z_e, self.num_codebooks, dim=1) #list of num_codebooks tensors, each (B,embedding_dim, 4, 4)
       
        quantized_chunks = []
        encoding_indices_list = []
        vq_loss_list = []
        
        for i, vq in enumerate(self.vq_layers):
            z_q, vq_loss, encoding_indices = vq(z_chunks[i])
            vq_loss_list.append(vq_loss)
            quantized_chunks.append(z_q)
            encoding_indices_list.append(encoding_indices)
        tokens = torch.stack(encoding_indices_list, dim=-1)  # [B, H/4, W/4, num_codebooks]
        return tokens  #[B, H/4, W/4, num_codebooks]

    # tokens to image
    # tokens [B, 4, 4, num_codebooks]
    # image is [B, 1, H, W]
    def decode(self, tokens):
        B, H, W, num_codebooks = tokens.shape
        quantized_latents = []

        for i, _ in enumerate(self.vq_layers):
            tokens_per_codebook = tokens[:, :, :, i]  # [B,H,W]
            flat_tokens = tokens_per_codebook.reshape(-1)  # [B*H*W]
            embeddings = self.vq_layers[i].embedding(flat_tokens)  # [B*H*W, embedding_dim]
            embeddings = embeddings.view(B, H, W, -1)
            quantized_latents.append(embeddings)
        quantized = torch.cat(quantized_latents, dim=-1) # [B, H, W, latent_dim]
        quantized = quantized.permute(0, 3, 1, 2)  # [B, latent_dim, H, W]
        return self.decoder(quantized) # [B, C, 4*H, 4*W]
    
