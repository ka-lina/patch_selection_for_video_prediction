# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionLoss(nn.Module):
    """
    Loss for video prediction.
    Combines MSE and optionally perceptual loss.
    """
    
    def __init__(self, use_l1=False, use_ssim=False):
        super().__init__()
        self.use_l1 = use_l1
        self.use_ssim = use_ssim
        
        if use_l1:
            self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, mask=None):
        """
        pred: (B, T, C, H, W) - predicted frames
        target: (B, T, C, H, W) - ground truth frames
        mask: (B, T, H, W) - optional mask for weighted loss
        """
        if mask is not None:
            pred = pred * mask.unsqueeze(2)
            target = target * mask.unsqueeze(2)
        
        # MSE loss (main)
        mse = self.mse_loss(pred, target)
        
        losses = {'mse': mse}
        total_loss = mse
        
        if self.use_l1:
            l1 = self.l1_loss(pred, target)
            losses['l1'] = l1
            total_loss = total_loss + 0.1 * l1
        
        if self.use_ssim:
            ssim = self._ssim_loss(pred, target)
            losses['ssim'] = ssim
            total_loss = total_loss + 0.1 * (1 - ssim)
        
        losses['total'] = total_loss
        return total_loss, losses
    
    def _ssim_loss(self, pred, target):
        """Simple SSIM approximation."""
        C = pred.shape[2]
        if C == 1:
            # Use a simple gradient-based similarity
            pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
            pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
            target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
            target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
            
            dx_sim = F.cosine_similarity(pred_dx.flatten(1), target_dx.flatten(1), dim=1).mean()
            dy_sim = F.cosine_similarity(pred_dy.flatten(1), target_dy.flatten(1), dim=1).mean()
            return (dx_sim + dy_sim) / 2
        return torch.tensor(0.5, device=pred.device)


class AutoGazePredictionLoss(nn.Module):
    """
    Combined loss for training AutoGaze with prediction objective.
    
    L = L_pred + λ_sparsity * L_sparsity + λ_entropy * L_entropy
    """
    
    def __init__(self, lambda_sparsity=0.01, lambda_entropy=0.001):
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_entropy = lambda_entropy
        self.pred_loss = PredictionLoss()
    
    def forward(self, pred, target, patch_mask, scores=None):
        """
        pred: (B, T_pred, C, H, W)
        target: (B, T_pred, C, H, W)
        patch_mask: (B, T_in, N) - binary mask for selected patches
        scores: (B, T_in, N) - importance scores (optional, for entropy)
        """
        B = pred.shape[0]
        
        # Prediction loss
        pred_loss, pred_losses = self.pred_loss(pred, target)
        
        # Sparsity loss (encourage fewer patches)
        avg_patches = patch_mask.sum(dim=-1).float().mean()
        sparsity_loss = avg_patches / patch_mask.shape[-1]
        
        # Entropy loss (encourage confident patch selection)
        entropy_loss = torch.tensor(0.0, device=pred.device)
        if scores is not None:
            probs = F.softmax(scores, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy_loss = -entropy  # Minimize negative entropy = maximize entropy
        
        # Combined loss
        total_loss = pred_loss + \
                     self.lambda_sparsity * sparsity_loss + \
                     self.lambda_entropy * entropy_loss
        
        return total_loss, {
            'pred_loss': pred_loss.item(),
            'sparsity': sparsity_loss.item(),
            'entropy': -entropy_loss.item(),
            'avg_patches': avg_patches.item(),
            **{f'pred_{k}': v.item() for k, v in pred_losses.items()},
        }