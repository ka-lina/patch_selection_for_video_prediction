# train_videomae.py - Updated with resume support
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math
from tqdm import tqdm
import os
import numpy as np

def train_videomae(config, train_loader, val_loader, model=None, start_epoch=0, optimizer_state=None):
    """
    Stage 1: Train the Predictive VideoMAE on MovingMNIST.
    
    Args:
        model: Existing model to continue training (if None, creates new model)
        start_epoch: Epoch to start from (for resuming)
        optimizer_state: Saved optimizer state (for resuming)
    """
    device = config.train_config.device
    
    # Create or use existing model
    if model is None:
        # from videomae import PredictiveVideoMAE
        from videomae_v2 import MemorySafeVideoMAE as PredictiveVideoMAE
        model = PredictiveVideoMAE(config.videomae_config)
        model = model.to(device)
    
    # Loss function
    from losses import PredictionLoss
    criterion = PredictionLoss(use_l1=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_config.videomae_lr,
        weight_decay=config.train_config.weight_decay,
    )
    
    # Load optimizer state if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("  Loaded optimizer state")
    
    # Learning rate scheduler
    total_epochs = config.train_config.videomae_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, total_epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred_frames = model(inputs, num_pred_frames=config.data_config.pred_frames)
                loss, loss_dict = criterion(pred_frames, targets)
            
            scaler.scale(loss).backward()
            
            if config.train_config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_config.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item() * inputs.shape[0]
            train_count += inputs.shape[0]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_dict["mse"].item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })
        
        scheduler.step()
        avg_train_loss = train_loss_sum / train_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                with torch.amp.autocast('cuda'):
                    pred_frames = model(inputs)
                    loss, _ = criterion(pred_frames, targets)
                
                val_loss_sum += loss.item() * inputs.shape[0]
                val_count += inputs.shape[0]
        
        avg_val_loss = val_loss_sum / val_count
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save checkpoint
        os.makedirs(config.train_config.save_dir, exist_ok=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(config.train_config.save_dir, 'videomae_best.pt'))
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
    
    return model, train_losses, val_losses