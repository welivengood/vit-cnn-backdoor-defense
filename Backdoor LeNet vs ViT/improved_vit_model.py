# vit_model.py
# Not the old version, stole from https://github.com/shub-garg/Vision-Transformer-VIT-for-MNIST/blob/main/Vision_Transformer_for_MNIST.ipynb

import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

# Applies layer normalization before passing through the given function (Attention or FeedForward)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Standard feedforward network used in transformers
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # Expands dimensionality
            nn.ReLU(),                   # Non-linearity (GELU often used in ViTs)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # Projects back to original dim
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Multi-head self-attention layer
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # Total embedding size after all heads
        project_out = not (heads == 1 and dim_head == dim)  # If we need a final projection

        self.heads = heads
        self.scale = dim_head ** -0.5  # Scaling factor for dot-product attention

        self.attend = nn.Softmax(dim = -1)  # Softmax over attention scores
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # Single linear layer to compute Q, K, V

        # Final projection after combining heads
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        ## STORES FOR GRAD-CAM ##
        self.last_attn = None  # store for later


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # batch, tokens, dim, num_heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # Split Q, K, V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # Reshape for multi-heads

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # QK^T
        attn = self.attend(dots)
        self.last_attn = attn 

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Weighted sum of values
        out = rearrange(out, 'b h n d -> b n (h d)')  # Recombine heads
        return self.to_out(out)

# Transformer encoder block: contains stacked attention + feedforward layers
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  # Self-attention
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))                            # Feed-forward
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual connection after attention
            x = ff(x) + x    # Residual connection after feedforward
        return x

# Full Vision Transformer model
class VITClassifier(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # Total number of patches
        patch_dim = channels * patch_height * patch_width  # Flattened size of each patch
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Converts image into a sequence of patch embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # Flatten patches
            nn.Linear(patch_dim, dim),  # Project patch to embedding dimension
        )

        # Learnable position embeddings and CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Add one for CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # Learnable CLS token
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # 'cls' or 'mean'
        self.to_latent = nn.Identity()  # Placeholder in case you want to add extra projection later

        # Final classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # Turn image into patch embeddings
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # Duplicate CLS token for each image
        x = torch.cat((cls_tokens, x), dim=1)  # Prepend CLS token to patch embeddings
        x += self.pos_embedding[:, :(n + 1)]  # Add position information
        x = self.dropout(x)

        x = self.transformer(x)  # Pass through Transformer encoder

        # Pooling: use CLS token or average of all tokens
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)  # Optional extra projection

        return self.mlp_head(x)  # Final classification
