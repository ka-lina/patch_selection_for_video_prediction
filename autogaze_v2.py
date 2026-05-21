# autogaze_v2.py - Redesigned AutoGaze that actually trains

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TrainableAutoGaze(nn.Module):
    """
    AutoGaze that learns a per-patch importance score and a global threshold.
    Patches with score > threshold are selected.
    This allows the model to learn HOW MANY patches to select.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = config.img_size // config.patch_size
        self.num_patches = self.grid_size ** 2
        self.max_patches = config.max_patches_per_frame
        
        # Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
        )
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, config.hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        
        # Importance network
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
        
        # Learnable threshold (initialized to select ~25% of patches)
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Will be learned!
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_scores=False):
        B, T, C, H, W = x.shape
        
        # Spatial encoding
        x_spatial = []
        for t in range(T):
            feat = self.spatial_encoder(x[:, t])
            x_spatial.append(feat)
        x_spatial = torch.stack(x_spatial, dim=2)
        
        # Temporal convolution
        x_temp = self.temporal_conv(x_spatial)
        
        # Flatten and compute scores
        x_flat = rearrange(x_temp, 'b d t h w -> b t (h w) d')
        scores = self.importance_net(x_flat).squeeze(-1)  # (B, T, N) in [0, 1]
        
        if self.training:
            # Use straight-through estimator for thresholding
            # This is differentiable!
            mask = self._threshold_with_gradient(scores)
        else:
            # Hard thresholding at inference
            mask = (scores > self.threshold).float()
            # If too few patches selected, take top-k
            min_patches = max(4, self.max_patches // 4)
            for b in range(B):
                for t in range(T):
                    if mask[b, t].sum() < min_patches:
                        _, top_idx = torch.topk(scores[b, t], min_patches)
                        mask[b, t, :] = 0
                        mask[b, t, top_idx] = 1
        
        with torch.no_grad():
            num_selected = mask.sum(dim=-1).float().mean()
        
        output = {
            'patch_mask': mask,
            'num_patches_selected': num_selected,
        }
        
        if return_scores:
            output['scores'] = scores
        
        return output
    
    def _threshold_with_gradient(self, scores):
        """
        Differentiable thresholding using straight-through estimator.
        
        Forward: hard threshold (mask = scores > threshold)
        Backward: gradient flows through sigmoid approximation
        """
        # Hard threshold for forward pass
        hard_mask = (scores > self.threshold).float()
        
        # Soft approximation for backward pass
        # Sigmoid with temperature around threshold
        temperature = 0.1
        soft_mask = torch.sigmoid((scores - self.threshold) / temperature)
        
        # Straight-through: use hard in forward, soft gradient in backward
        mask = hard_mask - soft_mask.detach() + soft_mask
        
        return mask