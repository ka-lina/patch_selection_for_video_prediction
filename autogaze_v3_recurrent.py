# autogaze_v3_recurrent.py - AutoGaze with temporal memory

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvLSTMCell(nn.Module):
    """Simple ConvLSTM cell for spatial memory."""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim,  # 4 gates: input, forget, cell, output
            kernel_size, 
            padding=padding
        )
    
    def forward(self, x, state=None):
        """
        x: (B, C, H, W)
        state: tuple of (h, c) each (B, hidden_dim, H, W)
        """
        B, C, H, W = x.shape
        
        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell and hidden state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


class RecurrentAutoGaze(nn.Module):
    """
    AutoGaze with temporal memory via ConvLSTM.
    
    The model maintains a hidden state that tracks which regions
    have already been covered, allowing it to avoid re-selecting
    redundant patches in subsequent frames.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.grid_size = config.img_size // config.patch_size
        self.num_patches = self.grid_size ** 2
        self.max_patches = config.max_patches_per_frame
        
        # Spatial encoder (same as before)
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
        
        # Temporal memory: ConvLSTM maintains state across frames
        self.conv_lstm = ConvLSTMCell(
            input_dim=128,
            hidden_dim=64,
            kernel_size=3,
        )
        
        # After ConvLSTM, process to get importance scores
        self.importance_net = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # 1 channel per patch
        )
        
        # Temporal difference encoder (captures motion between frames)
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # 2 channels: current frame + difference
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
        )
        
        # Fusion of spatial features, memory, and motion
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64 + 32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, prev_state=None, return_state=False, return_scores=False):
        """
        x: (B, T, C, H, W) - input frames
        
        If prev_state is provided, the ConvLSTM continues from that state.
        If return_state=True, returns the final state for next call.
        """
        B, T, C, H, W = x.shape
        
        all_masks = []
        all_scores = []
        
        # Initialize or use previous ConvLSTM state
        lstm_state = prev_state
        
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            
            # 1. Spatial features
            spatial_feat = self.spatial_encoder(frame)  # (B, 128, grid, grid)
            
            # 2. Motion features (difference from previous frame)
            if t > 0:
                prev_frame = x[:, t-1]
                diff = torch.abs(frame - prev_frame)
                motion_input = torch.cat([frame, diff], dim=1)  # (B, 2, H, W)
            else:
                # No motion for first frame
                motion_input = torch.cat([frame, torch.zeros_like(frame)], dim=1)
            motion_feat = self.motion_encoder(motion_input)  # (B, 32, grid, grid)
            
            # 3. Update ConvLSTM with current features
            lstm_out, lstm_state = self.conv_lstm(spatial_feat, lstm_state)
            # lstm_out: (B, 64, grid, grid) - contains memory of previous frames
            
            # 4. Fuse all features
            fused = torch.cat([spatial_feat, lstm_out, motion_feat], dim=1)  # (B, 128+64+32, grid, grid)
            scores = self.fusion(fused).squeeze(1)  # (B, grid, grid)
            scores = scores.reshape(B, self.num_patches)  # (B, N)
            
            # 5. Apply threshold
            if self.training:
                mask = self._threshold_with_gradient(scores)
            else:
                mask = (scores > self.threshold).float()
                # Ensure minimum patches
                min_patches = max(4, self.max_patches // 8)
                for b in range(B):
                    if mask[b].sum() < min_patches:
                        _, top_idx = torch.topk(scores[b], min_patches)
                        mask[b, :] = 0
                        mask[b, top_idx] = 1
            
            all_masks.append(mask)
            all_scores.append(scores)
        
        # Stack results
        patch_mask = torch.stack(all_masks, dim=1)  # (B, T, N)
        
        output = {
            'patch_mask': patch_mask,
            'num_patches_selected': patch_mask.sum(dim=-1).float().mean(),
        }
        
        if return_scores:
            output['scores'] = torch.stack(all_scores, dim=1)
        
        if return_state:
            output['lstm_state'] = lstm_state
        
        return output
    
    def _threshold_with_gradient(self, scores):
        """Differentiable thresholding."""
        temperature = 0.1
        hard_mask = (scores > self.threshold).float()
        soft_mask = torch.sigmoid((scores - self.threshold) / temperature)
        mask = hard_mask - soft_mask.detach() + soft_mask
        return mask
    
    def reset_state(self, batch_size=1, device='cuda'):
        """Reset ConvLSTM state for a new video."""
        return None  # Will be initialized on first forward pass