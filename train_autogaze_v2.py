# train_autogaze_v2.py - Fix the function signature
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math
from tqdm import tqdm
import os

import numpy as np


def train_autogaze_v2(config, train_loader, val_loader, videomae_model, model=None, start_epoch=0, optimizer_state=None):
    device = config.train_config.device
    
    # Freeze VideoMAE
    videomae_model.eval()
    for param in videomae_model.parameters():
        param.requires_grad = False
    
    # Create or load model
    if model is None:
        from autogaze_v2 import TrainableAutoGaze
        model = TrainableAutoGaze(config.autogaze_config)
        model = model.to(device)
    
    # Simple MSE loss
    criterion = nn.MSELoss()
    
    # Optimizer with separate LR for threshold
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'threshold' not in n], 'lr': config.train_config.autogaze_lr},
        {'params': [model.threshold], 'lr': config.train_config.autogaze_lr * 10},
    ], weight_decay=0.01)
    
    # Load optimizer state if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("  Loaded optimizer state")
    
    total_epochs = config.train_config.autogaze_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_val_sparsity = float('inf')
    train_losses = []
    val_losses = []
    train_sparsity = []
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss_sum = 0.0
        train_sparsity_sum = 0.0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f'AutoGaze Epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                gaze_output = model(inputs, return_scores=False)
                patch_mask = gaze_output['patch_mask']
                
                pred_frames = videomae_model.get_sparse_forward(
                    inputs, patch_mask,
                    num_pred_frames=config.data_config.pred_frames,
                )
                
                pred_loss = criterion(pred_frames, targets)
                avg_patches = patch_mask.sum(dim=-1).float().mean()
                total_patches = patch_mask.shape[-1]

                # Sparsity reward: reward using fewer patches
                # Scale the sparsity term relative to prediction loss so they're balanced
                sparsity_term = (avg_patches / total_patches) * pred_loss.detach()

                # Combined loss
                lambda_sparsity = 0.05  # Adjust this weight
                loss = pred_loss + lambda_sparsity * sparsity_term
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item() * inputs.shape[0]
            train_sparsity_sum += gaze_output['num_patches_selected'].item() * inputs.shape[0]
            train_count += inputs.shape[0]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{pred_loss.item():.4f}',
                'sparse': f'{sparsity_term.item():.4f}',
                'patches': f'{avg_patches.item():.1f}',
                'thresh': f'{model.threshold.item():.3f}',
            })
        
        scheduler.step()
        avg_train_loss = train_loss_sum / train_count
        avg_train_sparsity = train_sparsity_sum / train_count
        train_losses.append(avg_train_loss)
        train_sparsity.append(avg_train_sparsity)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_sparsity_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                with torch.amp.autocast('cuda'):
                    gaze_output = model(inputs)
                    patch_mask = gaze_output['patch_mask']
                    pred_frames = videomae_model.get_sparse_forward(inputs, patch_mask)
                    pred_loss = criterion(pred_frames, targets)
                    avg_patches = patch_mask.sum(dim=-1).float().mean()
                    total_patches = patch_mask.shape[-1]

                    # Sparsity reward: reward using fewer patches
                    # Scale the sparsity term relative to prediction loss so they're balanced
                    sparsity_term = (avg_patches / total_patches) * pred_loss.detach()
                    lambda_sparsity = 0.05  # Adjust this weight

                    # Encourage threshold to be high (select fewer patches)
                    # threshold_penalty = torch.relu(0.3 - model.threshold)  # Penalize if threshold < 0.3
                    loss = pred_loss + lambda_sparsity * sparsity_term #+ 0.01 * threshold_penalty
                
                val_loss_sum += loss.item() * inputs.shape[0]
                val_sparsity_sum += gaze_output['num_patches_selected'].item() * inputs.shape[0]
                val_count += inputs.shape[0]
        
        avg_val_loss = val_loss_sum / val_count
        avg_val_sparsity = val_sparsity_sum / val_count
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {avg_train_loss:.6f}, Patches: {avg_train_sparsity:.1f}, Thresh: {model.threshold.item():.3f}")
        print(f"  Val   - Loss: {avg_val_loss:.6f}, Patches: {avg_val_sparsity:.1f}")
        
        os.makedirs(config.train_config.save_dir, exist_ok=True)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': model.threshold.item(),
            }, os.path.join(config.train_config.save_dir, 'autogaze_last_epoch.pt'))
        print(f"  -> Saved last_epoch")

        if avg_val_sparsity < best_val_sparsity:
            best_val_sparsity = avg_val_sparsity
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': model.threshold.item(),
            }, os.path.join(config.train_config.save_dir, 'autogaze_best_sparsity.pt'))
            print(f"  -> Saved best sparsity model")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': model.threshold.item(),
            }, os.path.join(config.train_config.save_dir, 'autogaze_best.pt'))
            print(f"  -> Saved best model")
    
    return model, train_losses, val_losses, train_sparsity