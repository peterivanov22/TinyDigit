import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_length, latent_height, latent_width, num_classes, num_codebooks, d_model, n_heads, n_layers, dropout = 0.2):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.height_embedding = nn.Embedding(latent_height, d_model)
        self.width_embedding = nn.Embedding(latent_width, d_model)

        self.position_embedding = nn.Embedding(seq_length, d_model)

        self.class_embedding = nn.Embedding(num_classes, d_model)  # For class token if needed
        self.channel_embedding = nn.Embedding(num_codebooks, d_model)     # <--- NEW
        # Add dropout layer for embeddings sum
        self.embedding_dropout = nn.Dropout(dropout)

        self.num_codebooks = num_codebooks
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, norm_first=False, batch_first=True), num_layers=n_layers
        )

        #self.output_dropout = nn.Dropout(dropout)  
        self.fc_out = nn.Linear(d_model, vocab_size-num_classes) # dont want to predict token corresponding to class label

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def calculate_latent_indices(self, T, device):
        
        # Number of spatial positions in the grid
        num_spatial_positions = self.latent_height * self.latent_width 

        # Generate spatial position indices
        pos = torch.arange(num_spatial_positions, device = device)  # [0, 1, 2, ..., num_spatial_positions-1]

        # Calculate height and width indices
        height_idx = pos // self.latent_height  # Integer division to get row index
        width_idx = pos % self.latent_width    # Modulo operation to get column index

        # Repeat each index num_codebooks times to account for each codebook
        height_idx = height_idx.repeat_interleave(self.num_codebooks)  # [0, 0, ..., 0, 1, 1, ..., 1, ..., num_codebooks-1]
        width_idx = width_idx.repeat_interleave(self.num_codebooks)    # [0, 0, ..., 0, 1, 1, ..., 1, ..., num_codebooks-1]

        return height_idx[:T], width_idx[:T]
                 
    def forward(self, x):
        # Implement the forward pass for the transformer
        # Example: x = self.transformer_layer(x)
        B, T = x.shape


        x = self.token_embedding(x)  # (B, T, d_model)
        
        # cycles 01201201 (can stop at 0, 1 if T%num_codeboks != 0)
        channel_indices = (torch.arange(T, device=x.device) % self.num_codebooks).unsqueeze(0).expand(B, -1)  # [B, T]
        channel_embeddings = self.channel_embedding(channel_indices)
        #token_embeddings = token_embeddings + channel_embeddings   # (B, T, d_model)
        
       
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos_embeddings = self.position_embedding(positions)  # (1, T, d_model)
        

        #x = x + pos_embeddings  # (B, T, d_model)
        
        height_index, width_index = self.calculate_latent_indices(T, x.device)
        height_embeddings = self.height_embedding(height_index)
        width_embeddings = self.width_embedding(width_index)

        x = x + height_embeddings + width_embeddings

        #class_embeddings = self.class_embedding(class_labels).unsqueeze(1)  # (B, 1, d_model)
        #x = torch.cat([class_embeddings, x], dim=1)  # (B, T+1, d_model)


        # PyTorch expects True where masking is applied (future positions)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # upper-triangular
        memory = torch.zeros(B, 1, x.size(-1), device=x.device)         # Dummy memory 

        x = self.transformer(tgt=x, memory=memory, tgt_mask=causal_mask)  # (B, T, d_model)
        
        x = self.fc_out(x)  # (B, T, vocab_size-num_classes = num_embeddings)
        return x