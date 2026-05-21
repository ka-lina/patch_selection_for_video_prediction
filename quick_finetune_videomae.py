# quick_finetune_videomae.py - Run this before AutoGaze training

import torch
from tqdm import tqdm
import os

def finetune_videomae_with_masks(config, train_loader, videomae_model, num_epochs=5):
    """Quickly fine-tune VideoMAE to handle sparse/masked inputs."""
    device = config.train_config.device
    
    videomae_model.train()
    optimizer = torch.optim.AdamW(videomae_model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Fine-tune epoch {epoch+1}/{num_epochs}'):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Create random masks at different sparsity levels
            B, T, _, H, W = inputs.shape
            patch_size = config.videomae_config.patch_size
            grid_size = H // patch_size
            num_patches = grid_size * grid_size
            
            # Random sparsity: keep 10-50% of patches
            keep_ratio = 0.1 + 0.4 * torch.rand(1).item()
            rand_mask = torch.rand(B, T, num_patches, device=device) < keep_ratio
            rand_mask = rand_mask.float()
            
            optimizer.zero_grad()
            pred = videomae_model.get_sparse_forward(inputs, rand_mask)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Fine-tune epoch {epoch+1}: loss = {total_loss/len(train_loader):.6f}')
    
    # Save fine-tuned model
    torch.save({
        'model_state_dict': videomae_model.state_dict(),
    }, os.path.join(config.train_config.save_dir, 'videomae_finetuned.pt'))
    
    return videomae_model