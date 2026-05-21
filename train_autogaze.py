# train_autogaze.py - Updated with resume support
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math
from tqdm import tqdm
import os

import numpy as np

def train_autogaze(config, train_loader, val_loader, videomae_model, model=None, start_epoch=0, optimizer_state=None):
    """
    Stage 2: Train AutoGaze with prediction loss.
    
    Args:
        model: Existing AutoGaze model to continue training (if None, creates new model)
        start_epoch: Epoch to start from (for resuming)
        optimizer_state: Saved optimizer state (for resuming)
    """
    device = config.train_config.device
    
    # Freeze VideoMAE
    videomae_model.eval()
    for param in videomae_model.parameters():
        param.requires_grad = False
    
    # Create or use existing model
    if model is None:
        from autogaze import SimpleAutoGaze
        model = SimpleAutoGaze(config.autogaze_config)
        model = model.to(device)
    
    # Loss function
    from losses import AutoGazePredictionLoss
    criterion = AutoGazePredictionLoss(
        lambda_sparsity=0.01,
        lambda_entropy=0.001,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_config.autogaze_lr,
        weight_decay=config.train_config.weight_decay,
    )
    
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("  Loaded optimizer state")
    
    # Learning rate scheduler
    total_epochs = config.train_config.autogaze_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_sparsity = []
    
    for epoch in range(start_epoch, total_epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_pred_loss_sum = 0.0
        train_sparsity_sum = 0.0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f'AutoGaze Epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                gaze_output = model(inputs, return_scores=True)
                patch_mask = gaze_output['patch_mask']
                scores = gaze_output['scores']
                
                pred_frames = videomae_model.get_sparse_forward(
                    inputs, patch_mask,
                    num_pred_frames=config.data_config.pred_frames,
                )
                
                loss, loss_dict = criterion(pred_frames, targets, patch_mask, scores)
            
            scaler.scale(loss).backward()
            
            if config.train_config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_config.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item() * inputs.shape[0]
            train_pred_loss_sum += loss_dict['pred_loss'] * inputs.shape[0]
            train_sparsity_sum += loss_dict['avg_patches'] * inputs.shape[0]
            train_count += inputs.shape[0]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pred': f'{loss_dict["pred_loss"]:.4f}',
                'patches': f'{loss_dict["avg_patches"]:.1f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })
        
        scheduler.step()
        avg_train_loss = train_loss_sum / train_count
        avg_train_pred_loss = train_pred_loss_sum / train_count
        avg_train_sparsity = train_sparsity_sum / train_count
        train_losses.append(avg_train_loss)
        train_sparsity.append(avg_train_sparsity)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_pred_loss_sum = 0.0
        val_sparsity_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                with torch.amp.autocast('cuda'):
                    gaze_output = model(inputs)
                    patch_mask = gaze_output['patch_mask']
                    
                    pred_frames = videomae_model.get_sparse_forward(
                        inputs, patch_mask,
                        num_pred_frames=config.data_config.pred_frames,
                    )
                    
                    loss, loss_dict = criterion(pred_frames, targets, patch_mask)
                
                val_loss_sum += loss.item() * inputs.shape[0]
                val_pred_loss_sum += loss_dict['pred_loss'] * inputs.shape[0]
                val_sparsity_sum += loss_dict['avg_patches'] * inputs.shape[0]
                val_count += inputs.shape[0]
        
        avg_val_loss = val_loss_sum / val_count
        avg_val_pred_loss = val_pred_loss_sum / val_count
        avg_val_sparsity = val_sparsity_sum / val_count
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {avg_train_loss:.6f}, Pred: {avg_train_pred_loss:.6f}, Patches: {avg_train_sparsity:.1f}")
        print(f"  Val   - Loss: {avg_val_loss:.6f}, Pred: {avg_val_pred_loss:.6f}, Patches: {avg_val_sparsity:.1f}")
        
        os.makedirs(config.train_config.save_dir, exist_ok=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_sparsity': avg_train_sparsity,
            }, os.path.join(config.train_config.save_dir, 'autogaze_best.pt'))
            print(f"  -> Saved best model")
    
    return model, train_losses, val_losses, train_sparsity