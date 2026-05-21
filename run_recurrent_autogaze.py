# run_recurrent_autogaze.py

import torch
import argparse
from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.npy')
    parser.add_argument('--videomae_checkpoint', type=str, default='./checkpoints_v2/videomae_finetuned.pt')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_v2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--moving_bg', action='store_true', help='Use moving background dataset')
    args = parser.parse_args()
    
    device = 'cuda'
    
    # Configs
    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    from videomae_v2 import MemorySafeVideoMAE
    
    # Create configs
    data_config = DataConfig(
        dataset_path="mnist_test_seq.npy",
        input_frames=10,
        pred_frames=5,
        batch_size=16,
        use_moving_background=args.moving_bg,
    )

    videomae_config = VideoMAEConfig(
        img_size=data_config.img_size,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_depth=3,
        decoder_num_heads=4,
    )
    
    model = MemorySafeVideoMAE(videomae_config)
    autogaze_config = AutoGazeConfig(img_size=64, patch_size=4, max_patches_per_frame=64)
    
    train_config = TrainConfig(
        autogaze_lr=args.lr,
        autogaze_epochs=args.epochs,
        save_dir=args.save_dir,
        device=device,
    )
    
    class Config:
        pass
    config = Config()
    config.data_config = data_config
    config.videomae_config = videomae_config
    config.autogaze_config = autogaze_config
    config.train_config = train_config
    
    # Load VideoMAE
    from videomae_v2 import MemorySafeVideoMAE
    videomae = MemorySafeVideoMAE(videomae_config).to(device)
    ckpt = torch.load(args.videomae_checkpoint, map_location=device, weights_only=False)
    videomae.load_state_dict(ckpt['model_state_dict'])
    videomae.eval()
    print(f"Loaded VideoMAE from {args.videomae_checkpoint}")
    
    # Create dataloaders
    from dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Train recurrent AutoGaze
    from train_autogaze_v5_recurrent import train_autogaze_recurrent
    
    print("\n" + "="*60)
    print("Training Recurrent AutoGaze")
    print("="*60)
    
    model, train_losses, val_losses, train_sparsity = train_autogaze_recurrent(
        config, train_loader, val_loader, videomae,
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_sparsity': train_sparsity,
    }, os.path.join(args.save_dir, 'autogaze_recurrent_final.pt'))
    
    print(f"\nSaved to {args.save_dir}/autogaze_recurrent_final.pt")


if __name__ == '__main__':
    main()