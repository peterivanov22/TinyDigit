from PIL import Image
from torchvision import transforms
import torch

import torch.nn as nn
from torch.nn import functional as F


class GPTArtModel(nn.Module):
    
    def __init__(
        self,
        n_color = 256,
        n_embd = 256,
        block_size = 128
    ):
        super().__init__()
        self.red_embedding = nn.Embedding(256, n_color)
        self.green_embedding = nn.Embedding(256, n_color)
        self.blue_embedding = nn.Embedding(256, n_color)
        
        self.project_colors = nn.Linear(3*n_color, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, 3*256)

    def forward(self, idx, training=True):
        #idx has dimension (B,T,C=3)
        x_r = self.red_embedding(idx[:,:,0]) #(B,T,n_color)
        x_g = self.blue_embedding(idx[:,:,1]) #(B,T,n_color)
        x_b = self.green_embedding(idx[:,:,2]) #(B,T,n_color) 
        x = torch.cat([x_r, x_g, x_b], dim=-1) #(B,T,3*n_color)
        x = self.project_colors(x) #(B,T, n_embd)
        logits = self.lm_head(x) #(B,T, 3*256)

        return logits

    def generate(self, idx, max_new_tokens):
        #idx is (B,T,3) array of infixes in current context
        for _ in range(max_new_tokens):
            #get predictions
            logits, loss = self(idx)
            #focus on last time step
            logits = logits[:, -1, :, :] #becomes (B,E,C=3)
            logits_r = logits[:, :, 0] #(B,E)
            logits_g = logits[:, :, 1]
            logits_b = logits[:, :, 2]

            probs_r  = F.softmax(logits_r, dim=-1)
            probs_g = F.softmax(logits_g, dim=-1)
            probs_b = F.softmax(logits_b, dim=-1)
            probs_all = F.softmax(logits, dim = -2)
            #print(probs_all.shape) #(B, E, 3)


            #below we want to make guess of all 3 rbg values at once, not separately
            B, E, C = probs_all.shape
            temp_idx = []
            for i in range(B):
                curr_probs = probs_all[i]
                curr_probs = curr_probs.permute(1,0)
                temp_idx.append(torch.multinomial(curr_probs, num_samples=1))
            temp_idx = torch.stack(temp_idx)
            temp_idx = temp_idx.permute(0,-1,-2)
            #print(temp_idx.shape)

            idx_next_r = torch.multinomial(probs_r,  num_samples=1) # hopefully? (B,1) ???
            idx_next_g = torch.multinomial(probs_g,  num_samples=1) # hopefully? (B,1) ???
            idx_next_b = torch.multinomial(probs_b,  num_samples=1) # hopefully? (B,1) ???

            temp_r = torch.cat((idx[:,:,0], idx_next_r), dim=1) #?? (B,T+1, 3) ???
            temp_g = torch.cat((idx[:,:,1], idx_next_g), dim=1) #?? (B,T+1, 3) ???
            temp_b = torch.cat((idx[:,:,2], idx_next_b), dim=1) #?? (B,T+1, 3) ???


            idx = torch.cat((idx, temp_idx), dim=1)
            #print(idx.shape)

            #idx  = torch.stack([temp_r, temp_g, temp_b])
            #idx =  idx.permute(1,2,0) #back to (B, T+1, 3)

        return idx 

