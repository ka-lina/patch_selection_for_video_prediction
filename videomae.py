# videomae.py - Fixed decode method and related fixes

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math

class PatchEmbed(nn.Module):
    """Convert video frames to patch embeddings."""
    
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T, H', W')
        x = x.flatten(3)  # (B, embed_dim, T, N)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # x: (B, N, D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PredictiveVideoMAE(nn.Module):
    """
    Predictive VideoMAE for video prediction.
    
    Takes input_frames, encodes them, and predicts pred_frames future frames.
    Uses causal temporal masking in the encoder.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        self.grid_size = config.img_size // config.patch_size
        self.num_patches = self.grid_size ** 2  # e.g., 64 for 8x8 patches on 64x64
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        
        # Positional embeddings (learnable)
        max_frames = 50  # Support up to 50 frames
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, config.embed_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, max_frames, 1, config.embed_dim))
        
        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
            ) for _ in range(config.depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(config.embed_dim, config.embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, config.embed_dim))
        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, max_frames, 1, config.embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                num_heads=config.decoder_num_heads,
                mlp_ratio=config.mlp_ratio,
            ) for _ in range(config.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Output projection: patch_dim = patch_size * patch_size * in_channels
        self.patch_dim = config.patch_size * config.patch_size * config.in_channels
        self.decoder_pred = nn.Linear(config.embed_dim, self.patch_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_temp_embed, std=0.02)
    
    def encode(self, x):
        """
        Encode input frames.
        x: (B, T_in, C, H, W)
        Returns encoded features: (B, T_in * N, D)
        """
        B, T, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, T, N, D)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :, :, :]
        x = x + self.temp_embed[:, :T, :, :]
        
        # Flatten tokens
        x = rearrange(x, 'b t n d -> b (t n) d')
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x  # (B, T*N, D)
    
    def decode(self, encoded, num_pred_frames):
        """
        Decode encoded features to predict future frames.
        encoded: (B, T_in*N, D) where N = num_patches
        num_pred_frames: int - number of frames to predict
        Returns: (B, T_pred, C, H, W)
        """
        B = encoded.shape[0]
        D = encoded.shape[-1]
        T_enc = encoded.shape[1] // self.num_patches
        
        # Get a summary of all encoded information
        encoded_reshaped = rearrange(encoded, 'b (t n) d -> b t n d', t=T_enc, n=self.num_patches)
        encoded_summary = encoded_reshaped.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, D)
        
        # Create decoder tokens for each prediction frame
        decoder_tokens = encoded_summary.expand(-1, num_pred_frames, self.num_patches, -1).clone()
        
        # Add decoder positional embeddings
        decoder_tokens = decoder_tokens + self.decoder_pos_embed[:, :, :, :]
        decoder_tokens = decoder_tokens + self.decoder_temp_embed[:, :num_pred_frames, :, :]
        
        # Flatten for transformer
        x = rearrange(decoder_tokens, 'b t n d -> b (t n) d')
        
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Project to pixels
        x = self.decoder_pred(x)  # (B, T_pred * N, patch_dim)
        
        # Manual reshape (avoid einops issues)
        # x: (B, T_pred * N, patch_dim) where patch_dim = p1*p2*c
        # We want: (B, T_pred, C, H, W) where H = g1*p1, W = g2*p2
        x = x.reshape(B, num_pred_frames, self.num_patches, self.patch_size, self.patch_size, self.in_channels)
        
        # Reorder: (B, T_pred, N, p1, p2, c) -> we need N to become spatial grid
        # N = g1 * g2, so reshape: (B, T_pred, g1, g2, p1, p2, c)
        x = x.reshape(B, num_pred_frames, self.grid_size, self.grid_size, 
                    self.patch_size, self.patch_size, self.in_channels)
        
        # Combine spatial dims: (B, T_pred, g1, g2, p1, p2, c) -> (B, T_pred, g1*p1, g2*p2, c)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)  # (B, T_pred, c, g1, p1, g2, p2)
        x = x.reshape(B, num_pred_frames, self.in_channels, 
                    self.grid_size * self.patch_size, self.grid_size * self.patch_size)
        
        return x
    
    def forward(self, x, num_pred_frames=None):
        """
        Forward pass.
        x: (B, T_in, C, H, W) - input frames
        num_pred_frames: int - number of frames to predict
        """
        if num_pred_frames is None:
            num_pred_frames = getattr(self.config, 'pred_frames', 5)
        
        # Encode input frames
        encoded = self.encode(x)  # (B, T_in*N, D)
        
        # Decode to predict future frames
        pred_frames = self.decode(encoded, num_pred_frames)
        
        return pred_frames
    
    def get_sparse_forward(self, x, patch_mask, num_pred_frames=None):
        """
        Forward pass with sparse patch selection.
        x: (B, T_in, C, H, W)
        patch_mask: (B, T_in, N) - binary mask, 1 = keep this patch
        """
        if num_pred_frames is None:
            num_pred_frames = 5
        
        B, T, C, H, W = x.shape
        
        # Patch embedding (compute all patches, but mask after)
        x = self.patch_embed(x)  # (B, T, N, D)
        x = x + self.pos_embed[:, :, :, :]
        x = x + self.temp_embed[:, :T, :, :]
        
        # Apply mask - set masked patches to zero
        x = x * patch_mask.unsqueeze(-1)
        
        # Flatten and encode
        x = rearrange(x, 'b t n d -> b (t n) d')
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        # Decode
        pred_frames = self.decode(x, num_pred_frames)
        
        return pred_frames