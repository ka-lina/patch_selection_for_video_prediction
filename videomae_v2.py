# videomae_v2.py - Memory-safe version for 12GB

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2  # 256
        
        # 2D conv for per-frame processing (much less memory than 3D)
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Process each frame separately to save memory
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)  # (B*T, embed_dim, grid, grid)
        x = x.flatten(2)  # (B*T, embed_dim, N)
        x = x.permute(0, 2, 1)  # (B*T, N, embed_dim)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1):
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MemorySafeVideoMAE(nn.Module):
    """
    Memory-safe VideoMAE that processes frames in small groups.
    
    Strategy:
    1. Process each frame's 256 patches with spatial attention (small: 256²)
    2. Pool each frame to a single vector
    3. Process the 10 frame vectors with temporal attention (tiny: 10²)
    4. Decode with convolutional upsampling
    
    This avoids the massive 2560² attention matrix!
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = getattr(config, 'patch_size', 4)
        self.in_channels = config.in_channels
        self.embed_dim = getattr(config, 'embed_dim', 128)
        self.grid_size = config.img_size // self.patch_size  # 16
        self.num_patches = self.grid_size ** 2  # 256
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=self.patch_size,
            in_channels=config.in_channels,
            embed_dim=self.embed_dim,
        )
        
        # Spatial position embedding (per frame)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.embed_dim))
        
        # Spatial encoder (processes each frame independently)
        # 256 tokens, 256² attention = 65K - very manageable!
        spatial_depth = getattr(config, 'depth', 4)
        num_heads = getattr(config, 'num_heads', 4)
        mlp_ratio = getattr(config, 'mlp_ratio', 2.0)
        
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_ratio)
            for _ in range(spatial_depth)
        ])
        
        # Temporal aggregation: pool spatial features per frame
        self.temporal_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        self.temp_pos_embed = nn.Parameter(torch.zeros(1, 20, 1, self.embed_dim))
        
        # Temporal encoder (processes frame-level features)
        # Only 10 tokens, 10² attention = 100 - tiny!
        temporal_depth = 2
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_ratio)
            for _ in range(temporal_depth)
        ])
        
        # Decoder: embed_dim -> grid -> img_size
        up1_dim = 64
        up2_dim = 32
        
        self.decoder_up1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, up1_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(up1_dim),
            nn.ReLU(),
        )
        
        self.decoder_up2 = nn.Sequential(
            nn.ConvTranspose2d(up1_dim, up2_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(up2_dim),
            nn.ReLU(),
        )
        
        pred_frames = getattr(config, 'pred_frames', 5)
        out_channels = config.in_channels * pred_frames
        self.final_conv = nn.Sequential(
            nn.Conv2d(up2_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_token, std=0.02)
        nn.init.trunc_normal_(self.temp_pos_embed, std=0.02)
    
    def encode(self, x):
        """Encode video frames into a compact representation."""
        B, T, C, H, W = x.shape  # We have B and T from the input!
        
        # Patch embedding: per-frame 2D conv
        x = self.patch_embed(x)  # (B, T, N, D)
        x = x + self.pos_embed[:, :, :, :]
        
        # Spatial encoding: process each frame's patches independently
        # Reshape to (B*T, N, D) to share computation
        x = rearrange(x, 'b t n d -> (b t) n d')
        for block in self.spatial_blocks:
            x = block(x)  # (B*T, N, D)
        
        # Global average pool per frame: (B*T, N, D) -> (B*T, D)
        x_spatial_pooled = x.mean(dim=1)  # (B*T, D)
        x_spatial_pooled = rearrange(x_spatial_pooled, '(b t) d -> b t d', b=B, t=T)
        
        # Save spatial features for decoder (reshape back to 2D)
        # Need to provide b=B and t=T explicitly
        x_spatial_2d = rearrange(x, '(b t) (g1 g2) d -> (b t) d g1 g2',
                                b=B, t=T, g1=self.grid_size, g2=self.grid_size)
        
        # Temporal encoding: process the 10 frame-level features
        temporal_tokens = x_spatial_pooled.unsqueeze(2)  # (B, T, 1, D)
        temporal_tokens = temporal_tokens + self.temporal_token[:, :, :, :]
        temporal_tokens = temporal_tokens + self.temp_pos_embed[:, :T, :, :]
        temporal_tokens = rearrange(temporal_tokens, 'b t n d -> (b t) n d')
        
        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens)
        
        # Pool temporal features
        temporal_tokens = rearrange(temporal_tokens, '(b t) n d -> b t n d', b=B, t=T)
        temporal_pooled = temporal_tokens.mean(dim=1).squeeze(1)  # (B, D)
        
        # Combine spatial and temporal
        temporal_modulation = temporal_pooled.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        temporal_modulation = temporal_modulation.expand(-1, -1, self.grid_size, self.grid_size)
        
        # Average spatial features over time
        x_spatial_2d = rearrange(x_spatial_2d, '(b t) d g1 g2 -> b t d g1 g2', b=B, t=T)
        x_spatial_2d = x_spatial_2d.mean(dim=1)  # (B, D, grid, grid)
        
        # Fuse
        x = x_spatial_2d + temporal_modulation
        
        return x
    
    def decode(self, features, num_pred_frames):
        """Decode features to predicted frames."""
        B = features.shape[0]
        
        x = self.decoder_up1(features)   # (B, 64, 32, 32)
        x = self.decoder_up2(x)          # (B, 32, 64, 64)
        x = self.final_conv(x)           # (B, T_pred*C, 64, 64)
        x = x.reshape(B, num_pred_frames, self.in_channels, 
                      self.img_size, self.img_size)
        
        return x
    
    def forward(self, x, num_pred_frames=None):
        if num_pred_frames is None:
            num_pred_frames = getattr(self.config, 'pred_frames', 5)
        
        encoded = self.encode(x)
        pred_frames = self.decode(encoded, num_pred_frames)
        
        return pred_frames
    
    def get_sparse_forward(self, x, patch_mask, num_pred_frames=None):
        """Forward with sparse patches."""
        if num_pred_frames is None:
            num_pred_frames = 5
        
        B, T, C, H, W = x.shape  # We have B and T!
        
        # Patch embedding with mask
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :, :, :]
        x = x * patch_mask.unsqueeze(-1)
        
        # Spatial encoding
        x = rearrange(x, 'b t n d -> (b t) n d')
        for block in self.spatial_blocks:
            x = block(x)
        
        # Pool
        x_spatial_pooled = x.mean(dim=1)
        x_spatial_pooled = rearrange(x_spatial_pooled, '(b t) d -> b t d', b=B, t=T)
        
        x_spatial_2d = rearrange(x, '(b t) (g1 g2) d -> (b t) d g1 g2',
                                b=B, t=T, g1=self.grid_size, g2=self.grid_size)
        
        # Temporal encoding
        temporal_tokens = x_spatial_pooled.unsqueeze(2)
        temporal_tokens = temporal_tokens + self.temporal_token[:, :, :, :]
        temporal_tokens = temporal_tokens + self.temp_pos_embed[:, :T, :, :]
        temporal_tokens = rearrange(temporal_tokens, 'b t n d -> (b t) n d')
        
        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens)
        
        temporal_tokens = rearrange(temporal_tokens, '(b t) n d -> b t n d', b=B, t=T)
        temporal_pooled = temporal_tokens.mean(dim=1).squeeze(1)
        
        temporal_modulation = temporal_pooled.unsqueeze(-1).unsqueeze(-1)
        temporal_modulation = temporal_modulation.expand(-1, -1, self.grid_size, self.grid_size)
        
        x_spatial_2d = rearrange(x_spatial_2d, '(b t) d g1 g2 -> b t d g1 g2', b=B, t=T)
        x_spatial_2d = x_spatial_2d.mean(dim=1)
        
        x = x_spatial_2d + temporal_modulation
        
        pred_frames = self.decode(x, num_pred_frames)
        return pred_frames

    def get_sparse_forward_efficient(self, x, patch_mask, num_pred_frames=None):
        """
        Process only selected patches through spatial encoder.
        Still uses fixed grid for decoder (no architectural change needed).
        """
        if num_pred_frames is None:
            num_pred_frames = 5
        
        B, T, C, H, W = x.shape
        
        # 1. Get patch embeddings for ALL patches (lightweight conv)
        patches = self.patch_embed(x)  # (B, T, N, D)
        patches = patches + self.pos_embed[:, :, :, :]
        patches = patches + self.temp_embed[:, :T, :, :]
        
        # 2. Reshape for per-frame processing
        patches = rearrange(patches, 'b t n d -> (b t) n d')
        mask_flat = rearrange(patch_mask, 'b t n -> (b t) n')
        
        # 3. Process only selected patches through spatial encoder
        # This is where we save computation!
        BT, N, D = patches.shape
        outputs = torch.zeros(BT, N, D, device=patches.device)
        
        for i in range(BT):
            selected_idx = mask_flat[i].bool()  # (N,)
            if selected_idx.sum() > 0:
                selected_patches = patches[i, selected_idx, :]  # (k, D)
                # Add a CLS token for global context
                cls_token = selected_patches.mean(dim=0, keepdim=True)  # (1, D)
                selected_with_cls = torch.cat([cls_token, selected_patches], dim=0)  # (k+1, D)
                
                # Run through spatial transformer blocks (smaller sequence!)
                for block in self.spatial_blocks:
                    selected_with_cls = block(selected_with_cls)
                
                # Scatter back to full grid
                outputs[i, selected_idx, :] = selected_with_cls[1:, :]  # Remove CLS
        
        # 4. Rest of pipeline stays the same (fixed grid operations)
        # Spatial encoding
        x = rearrange(x, 'b t n d -> (b t) n d')
        for block in self.spatial_blocks:
            x = block(x)
        
        # Pool
        x_spatial_pooled = x.mean(dim=1)
        x_spatial_pooled = rearrange(x_spatial_pooled, '(b t) d -> b t d', b=B, t=T)
        
        x_spatial_2d = rearrange(x, '(b t) (g1 g2) d -> (b t) d g1 g2',
                                b=B, t=T, g1=self.grid_size, g2=self.grid_size)
        
        # Temporal encoding
        temporal_tokens = x_spatial_pooled.unsqueeze(2)
        temporal_tokens = temporal_tokens + self.temporal_token[:, :, :, :]
        temporal_tokens = temporal_tokens + self.temp_pos_embed[:, :T, :, :]
        temporal_tokens = rearrange(temporal_tokens, 'b t n d -> (b t) n d')
        
        for block in self.temporal_blocks:
            temporal_tokens = block(temporal_tokens)
        
        temporal_tokens = rearrange(temporal_tokens, '(b t) n d -> b t n d', b=B, t=T)
        temporal_pooled = temporal_tokens.mean(dim=1).squeeze(1)
        
        temporal_modulation = temporal_pooled.unsqueeze(-1).unsqueeze(-1)
        temporal_modulation = temporal_modulation.expand(-1, -1, self.grid_size, self.grid_size)
        
        x_spatial_2d = rearrange(x_spatial_2d, '(b t) d g1 g2 -> b t d g1 g2', b=B, t=T)
        x_spatial_2d = x_spatial_2d.mean(dim=1)
        
        x = x_spatial_2d + temporal_modulation
        
        pred_frames = self.decode(x, num_pred_frames)
        return pred_frames