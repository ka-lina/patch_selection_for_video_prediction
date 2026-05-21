# finetune_videomae.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import argparse

def finetune_videomae_with_masks(videomae_model, train_loader, val_loader, save_dir, num_epochs=5, lr=1e-5, device='cuda'):
    """Fine-tune VideoMAE to handle sparse/masked inputs."""
    
    # Load model if path provided
    if isinstance(videomae_model, str):
        from videomae_v2 import MemorySafeVideoMAE
        from config import VideoMAEConfig
        config = VideoMAEConfig()
        videomae_model = MemorySafeVideoMAE(config)
        checkpoint = torch.load(videomae_model, map_location=device)
        videomae_model.load_state_dict(checkpoint['model_state_dict'])
    
    videomae_model = videomae_model.to(device)
    videomae_model.train()
    
    optimizer = torch.optim.AdamW(videomae_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Fixed parameters
    patch_size = 4
    grid_size = 64 // patch_size  # 16
    num_patches = grid_size * grid_size  # 256
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Fine-tune epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            inputs = batch['input'].to(device)   # (B, T, 1, 64, 64)
            targets = batch['target'].to(device)  # (B, T_pred, 1, 64, 64)
            
            B, T, C, H, W = inputs.shape
            
            # Vary sparsity: keep between 10% and 50% of patches
            keep_ratio = 0.1 + 0.4 * torch.rand(1).item()
            rand_mask = torch.rand(B, T, num_patches, device=device) < keep_ratio
            rand_mask = rand_mask.float()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = videomae_model.get_sparse_forward(inputs, rand_mask)
                loss = criterion(pred, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'keep%': f'{keep_ratio:.1%}'})
        
        avg_loss = total_loss / max(total_batches, 1)
        print(f'Fine-tune epoch {epoch+1}: avg loss = {avg_loss:.6f}')
        
        # Quick validation
        videomae_model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch['input'].to(device)
                val_targets = val_batch['target'].to(device)
                
                B_val = val_inputs.shape[0]
                
                # Create mask for THIS batch size
                val_mask = torch.rand(B_val, T, num_patches, device=device) < 0.25
                val_mask = val_mask.float()
                
                with torch.amp.autocast('cuda'):
                    val_pred = videomae_model.get_sparse_forward(val_inputs, val_mask)
                    val_loss += criterion(val_pred, val_targets).item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f'Validation loss (25% mask): {avg_val_loss:.6f}')
        videomae_model.train()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'videomae_finetuned.pt')
    torch.save({'model_state_dict': videomae_model.state_dict()}, save_path)
    print(f'Saved fine-tuned model to {save_path}')
    
    return videomae_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videomae_checkpoint', type=str, default='./checkpoints_v2/videomae_best.pt')
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.npy')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_v2')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    from videomae_v2 import MemorySafeVideoMAE
    from config import VideoMAEConfig, DataConfig
    from dataset import MovingMNISTDataset, DataLoader
    
    # Load model
    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    
    # Create configs
    data_config = DataConfig(
        dataset_path=args.data_path,
        input_frames=10,
        pred_frames=5,
        batch_size=args.batch_size,
        use_moving_background=False,
    )
    
    config = VideoMAEConfig(
        img_size=data_config.img_size,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_depth=3,
        decoder_num_heads=4,
    )
    model = MemorySafeVideoMAE(config)
    checkpoint = torch.load(args.videomae_checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    train_dataset = MovingMNISTDataset(args.data_path, input_frames=10, pred_frames=5, train=True)
    val_dataset = MovingMNISTDataset(args.data_path, input_frames=10, pred_frames=5, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    finetune_videomae_with_masks(model, train_loader, val_loader, args.save_dir, args.epochs, args.lr)