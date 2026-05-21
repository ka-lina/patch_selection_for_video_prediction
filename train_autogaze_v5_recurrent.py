# train_autogaze_v3.py - Complete training for recurrent AutoGaze

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

def train_autogaze_recurrent(config, train_loader, val_loader, videomae_model, 
                             model=None, start_epoch=0, optimizer_state=None):
    """
    Train recurrent AutoGaze with temporal memory.
    """
    device = config.train_config.device
    
    # Freeze VideoMAE
    videomae_model.eval()
    for param in videomae_model.parameters():
        param.requires_grad = False
    
    # Create or load model
    if model is None:
        from autogaze_v3_recurrent import RecurrentAutoGaze
        model = RecurrentAutoGaze(config.autogaze_config)
        model = model.to(device)
        print(f"Created RecurrentAutoGaze with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Loss
    criterion = nn.MSELoss()
    
    # Optimizer with separate LR for threshold
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'threshold' not in n], 
         'lr': config.train_config.autogaze_lr},
        {'params': [model.threshold], 
         'lr': config.train_config.autogaze_lr * 10},  # Threshold learns faster
    ], weight_decay=0.01)
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("  Loaded optimizer state")
    
    total_epochs = config.train_config.autogaze_epochs
    
    # LR scheduler with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_sparsity = []
    
    # Baseline loss for quality tracking
    baseline_loss = 0.035  # Full model MSE on this data
    
    for epoch in range(start_epoch, total_epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_pred_loss_sum = 0.0
        train_sparsity_sum = 0.0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f'Recurrent AG Epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            inputs = batch['input'].to(device)     # (B, T, 1, H, W)
            targets = batch['target'].to(device)    # (B, T_pred, 1, H, W)
            B, T = inputs.shape[:2]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward pass - model maintains state across frames internally
                gaze_output = model(inputs, return_state=False, return_scores=True)
                patch_mask = gaze_output['patch_mask']  # (B, T, N)
                scores = gaze_output.get('scores', None)
                
                # VideoMAE prediction with selected patches
                pred_frames = videomae_model.get_sparse_forward(
                    inputs, patch_mask,
                    num_pred_frames=config.data_config.pred_frames,
                )
                
                # Prediction loss
                pred_loss = criterion(pred_frames, targets)
                
                # Sparsity
                avg_patches = patch_mask.sum(dim=-1).float().mean()
                total_patches = patch_mask.shape[-1]
                sparsity_ratio = avg_patches / total_patches
                
                # Quality budget: allow 10% degradation from baseline
                quality_target = baseline_loss * 1.10
                quality_penalty = torch.relu(pred_loss - quality_target)
                
                # Combined loss
                alpha = 1.0    # Sparsity weight
                beta = 10.0    # Quality constraint weight
                loss = alpha * sparsity_ratio + beta * quality_penalty
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item() * B
            train_pred_loss_sum += pred_loss.item() * B
            train_sparsity_sum += avg_patches.item() * B
            train_count += B
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}',
                'patches': f'{avg_patches.item():.1f}',
                'sparse%': f'{sparsity_ratio.item():.1%}',
                'thresh': f'{model.threshold.item():.3f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })
        
        scheduler.step()
        avg_train_loss = train_loss_sum / train_count
        avg_train_pred = train_pred_loss_sum / train_count
        avg_train_sparsity = train_sparsity_sum / train_count
        train_losses.append(avg_train_loss)
        train_sparsity.append(avg_train_sparsity)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_pred_sum = 0.0
        val_sparsity_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                B = inputs.shape[0]
                
                with torch.amp.autocast('cuda'):
                    gaze_output = model(inputs, return_state=False)
                    patch_mask = gaze_output['patch_mask']
                    
                    pred_frames = videomae_model.get_sparse_forward(inputs, patch_mask)
                    pred_loss = criterion(pred_frames, targets)
                    
                    avg_patches = patch_mask.sum(dim=-1).float().mean()
                    sparsity_ratio = avg_patches / patch_mask.shape[-1]
                    
                    quality_penalty = torch.relu(pred_loss - baseline_loss * 1.10)
                    val_loss = sparsity_ratio + 10.0 * quality_penalty
                
                val_loss_sum += val_loss.item() * B
                val_pred_sum += pred_loss.item() * B
                val_sparsity_sum += avg_patches.item() * B
                val_count += B
        
        avg_val_loss = val_loss_sum / val_count
        avg_val_pred = val_pred_sum / val_count
        avg_val_sparsity = val_sparsity_sum / val_count
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Pred: {avg_train_pred:.4f}, "
              f"Patches: {avg_train_sparsity:.1f}, Thresh: {model.threshold.item():.3f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Pred: {avg_val_pred:.4f}, "
              f"Patches: {avg_val_sparsity:.1f}")
        
        # Save checkpoint
        os.makedirs(config.train_config.save_dir, exist_ok=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': model.threshold.item(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_sparsity': avg_train_sparsity,
            }, os.path.join(config.train_config.save_dir, 'autogaze_recurrent_best.pt'))
            print(f"  -> Saved best model")
    
    return model, train_losses, val_losses, train_sparsity