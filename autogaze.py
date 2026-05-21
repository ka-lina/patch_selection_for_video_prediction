# autogaze.py - Fixed SimpleAutoGaze forward method

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List

class SimpleAutoGaze(nn.Module):
    """
    Simplified AutoGaze module for video prediction.
    Uses CNN + Gumbel-softmax for differentiable patch selection.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = config.img_size // config.patch_size
        self.num_patches = self.grid_size ** 2
        self.max_patches = config.max_patches_per_frame
        
        # CNN encoder for spatial features
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
        )
        
        # Temporal convolution to capture motion
        self.temporal_conv = nn.Conv3d(
            128, config.hidden_dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        
        # MLP for patch importance scoring
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        self.temperature = config.temperature
    
    def forward(self, x, return_scores=False):
        """
        x: (B, T, C, H, W) - input frames
        
        Returns dict with:
            patch_mask: (B, T, N) - binary mask for selected patches
            selected_indices: (B, T, max_patches) - indices of selected patches
            num_patches_selected: scalar - average number of patches selected
            scores: (B, T, N) - importance scores (if return_scores=True)
        """
        B, T, C, H, W = x.shape
        
        # Spatial encoding per frame
        x_spatial = []
        for t in range(T):
            feat = self.spatial_encoder(x[:, t])  # (B, 128, grid_size, grid_size)
            x_spatial.append(feat)
        x_spatial = torch.stack(x_spatial, dim=2)  # (B, 128, T, grid_size, grid_size)
        
        # Temporal convolution
        x_temp = self.temporal_conv(x_spatial)  # (B, hidden_dim, T, grid_size, grid_size)
        
        # Flatten spatial dimensions
        x_flat = rearrange(x_temp, 'b d t h w -> b t (h w) d')  # (B, T, N, D)
        
        # Compute importance scores
        scores = self.importance_net(x_flat).squeeze(-1)  # (B, T, N)
        
        # Store raw scores if needed
        raw_scores = scores.clone() if return_scores else None
        
        # Apply Gumbel-softmax for differentiable patch selection
        if self.training:
            patch_mask = self._gumbel_softmax_mask(scores, self.temperature)
        else:
            patch_mask = self._hard_selection(scores, self.max_patches)
        
        # Get selected patch indices (for visualization)
        with torch.no_grad():
            _, selected_indices = torch.topk(scores, self.max_patches, dim=-1)
            selected_indices = selected_indices.sort(dim=-1)[0]
        
        output = {
            'patch_mask': patch_mask,  # (B, T, N)
            'selected_indices': selected_indices,  # (B, T, max_patches)
            'num_patches_selected': patch_mask.sum(dim=-1).mean(),  # scalar
        }
        
        if return_scores:
            output['scores'] = raw_scores
        
        return output
    
    def _gumbel_softmax_mask(self, logits, temperature):
        """Gumbel-softmax for differentiable top-k selection."""
        B, T, N = logits.shape
        
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y = (logits + gumbel_noise) / temperature
        
        # Softmax
        y_soft = F.softmax(y, dim=-1)
        
        # Straight-through: hard selection in forward, soft gradients in backward
        _, top_indices = torch.topk(y_soft, self.max_patches, dim=-1)
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, top_indices, 1.0)
        
        # Straight-through estimator
        mask = y_hard - y_soft.detach() + y_soft
        
        return mask
    
    def _hard_selection(self, scores, k):
        """Hard top-k selection during inference."""
        B, T, N = scores.shape
        _, top_indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_indices, 1.0)
        return mask


class HeuristicPatchSelector:
    """
    Heuristic patch selection methods for comparison.
    """
    
    def __init__(self, method='difference', num_patches=64, patch_size=4):
        self.method = method
        self.num_patches = num_patches
        self.patch_size = patch_size
    
    def select_patches(self, frames):
        """
        frames: (B, T, C, H, W)
        Returns: (B, T, N) binary mask
        """
        B, T, C, H, W = frames.shape
        
        grid_size = H // self.patch_size
        num_patches_total = grid_size * grid_size
        k = min(self.num_patches, num_patches_total)
        
        mask = torch.zeros(B, T, num_patches_total, device=frames.device)
        
        if self.method == 'random':
            for b in range(B):
                for t in range(T):
                    indices = torch.randperm(num_patches_total, device=frames.device)[:k]
                    mask[b, t, indices] = 1.0
        
        elif self.method == 'center':
            center_h, center_w = grid_size // 2, grid_size // 2
            distances = []
            for h in range(grid_size):
                for w in range(grid_size):
                    dist = (h - center_h) ** 2 + (w - center_w) ** 2
                    distances.append(dist)
            distances = torch.tensor(distances, device=frames.device)
            _, top_indices = torch.topk(-distances, k, dim=0)
            mask[:, :, top_indices] = 1.0
        
        elif self.method == 'difference':
            for t in range(T):
                if t == 0:
                    frame = frames[:, t, 0]
                    patch_h = H // grid_size
                    patch_w = W // grid_size
                    patches = frame.reshape(B, grid_size, patch_h, grid_size, patch_w)
                    patches = patches.permute(0, 1, 3, 2, 4)
                    patches = patches.reshape(B, num_patches_total, -1)
                    activity = patches.abs().mean(dim=-1)
                else:
                    diff = (frames[:, t] - frames[:, t-1]).abs()
                    diff = diff.mean(dim=1)
                    diff = F.avg_pool2d(diff.unsqueeze(1), 
                                       kernel_size=self.patch_size, 
                                       stride=self.patch_size)
                    activity = diff.flatten(1)
                
                _, top_indices = torch.topk(activity, k, dim=-1)
                for b in range(B):
                    mask[b, t, top_indices[b]] = 1.0
        
        elif self.method == 'grid':
            step = max(1, int((num_patches_total / k) ** 0.5))
            for h in range(0, grid_size, step):
                for w in range(0, grid_size, step):
                    idx = h * grid_size + w
                    if idx < num_patches_total:
                        mask[:, :, idx] = 1.0
            current = mask.sum(dim=-1).max().item()
            if current > k:
                active = mask[0, 0].nonzero(as_tuple=True)[0]
                mask[:, :, active[k:]] = 0.0
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return mask