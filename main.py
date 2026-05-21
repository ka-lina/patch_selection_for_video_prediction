# main.py - Updated with resume capability

import torch
import os
import sys
import argparse
import numpy as np
import random

from visualize import visualize_predictions, visualize_patch_selection, create_prediction_only_gif, create_multi_method_comparison_gif, create_patch_selection_gif

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # Import config
    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    
    # Create configs
    data_config = DataConfig(
        dataset_path=args.data_path,
        input_frames=args.input_frames,
        pred_frames=args.pred_frames,
        batch_size=args.batch_size,
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
    
    autogaze_config = AutoGazeConfig(
        img_size=data_config.img_size,
        patch_size=videomae_config.patch_size,  # 4 - match VideoMAE!
        max_patches_per_frame=args.max_patches,  # Default 64 (25% of 256)
    )
    
    train_config = TrainConfig(
        videomae_lr=args.videomae_lr,
        videomae_epochs=args.videomae_epochs,
        autogaze_lr=args.autogaze_lr,
        autogaze_epochs=args.autogaze_epochs,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Combine into a namespace
    class Config:
        pass
    config = Config()
    config.data_config = data_config
    config.videomae_config = videomae_config
    config.autogaze_config = autogaze_config
    config.train_config = train_config
    
    # Set seed
    set_seed(train_config.seed)
    
    # Create dataloaders
    from dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ================================================================
    # Stage 1: Train VideoMAE
    # ================================================================
    if not args.skip_videomae:
        print("\n" + "="*60)
        print("STAGE 1: Training Predictive VideoMAE")
        print("="*60)
        
        from train_videomae import train_videomae
        from evaluate import evaluate_videomae_model as evaluate_videomae
        # from videomae import PredictiveVideoMAE
        from videomae_v2 import MemorySafeVideoMAE as PredictiveVideoMAE
        
        videomae_model = PredictiveVideoMAE(videomae_config)
        videomae_model = videomae_model.to(train_config.device)
        
        # Check for checkpoint to resume from
        videomae_path = os.path.join(train_config.save_dir, 'videomae_best.pt')
        start_epoch = 0
        optimizer_state = None
        
        if args.resume_videomae and os.path.exists(videomae_path):
            print(f"Resuming VideoMAE from {videomae_path}")
            checkpoint = torch.load(videomae_path, map_location=train_config.device, weights_only=False)
            videomae_model.load_state_dict(checkpoint['model_state_dict'])
            # finetuned_path = os.path.join(train_config.save_dir, 'videomae_finetuned.pt')
            # if os.path.exists(finetuned_path) and not args.skip_videomae:
            #     checkpoint = torch.load(finetuned_path, map_location=device)
            #     videomae_model.load_state_dict(checkpoint['model_state_dict'])
            #     print("Loaded fine-tuned VideoMAE")
            start_epoch = checkpoint.get('epoch', 0) + 1
            if 'optimizer_state_dict' in checkpoint:
                optimizer_state = checkpoint['optimizer_state_dict']
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Previous val loss: {checkpoint.get('val_loss', 'N/A')}")
        elif os.path.exists(videomae_path) and not args.force_retrain:
            print(f"Loading pretrained VideoMAE from {videomae_path}")
            checkpoint = torch.load(videomae_path, map_location=train_config.device, weights_only=False)
            videomae_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        else:
            print("Training VideoMAE from scratch")
        
        # Only train if we haven't finished all epochs
        if start_epoch < train_config.videomae_epochs:
            if start_epoch > 0:
                print(f"Continuing training from epoch {start_epoch} to {train_config.videomae_epochs}")
                train_config.videomae_epochs_remaining = train_config.videomae_epochs - start_epoch
            else:
                train_config.videomae_epochs_remaining = train_config.videomae_epochs
            
            videomae_model, train_losses, val_losses = train_videomae(
                config, train_loader, val_loader,
                model=videomae_model,  # Pass existing model
                start_epoch=start_epoch,
                optimizer_state=optimizer_state,
            )
            
            # Plot training curves
            from visualize import plot_training_curves
            plot_training_curves(
                train_losses, val_losses,
                save_path=os.path.join(train_config.save_dir, 'videomae_curves.png'),
            )
        else:
            print(f"VideoMAE already trained for {train_config.videomae_epochs} epochs, skipping training")
        
        # Evaluate VideoMAE baseline
        print("\nEvaluating VideoMAE baseline...")
        # from train_videomae import evaluate_videomae
        from evaluate import evaluate_videomae_model as evaluate_videomae
        metrics, all_preds, all_targets = evaluate_videomae(
            videomae_model, val_loader, train_config.device,
        )
        
        print("\nVideoMAE Baseline Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
    
    else:
        # Load pretrained VideoMAE
        # videomae_path = os.path.join(train_config.save_dir, 'videomae_best.pt')
        # if not os.path.exists(videomae_path):
        #     raise FileNotFoundError(f"No VideoMAE checkpoint at {videomae_path}. Run without --skip_videomae first.")
        
        # # from videomae import PredictiveVideoMAE
        from videomae_v2 import MemorySafeVideoMAE as PredictiveVideoMAE
        # checkpoint = torch.load(videomae_path, map_location=train_config.device, weights_only=False)

        finetuned_path = os.path.join(train_config.save_dir, 'videomae_finetuned.pt')
        if os.path.exists(finetuned_path):
            checkpoint = torch.load(finetuned_path, map_location=train_config.device, weights_only=False)
            
        videomae_model = PredictiveVideoMAE(videomae_config)
        videomae_model.load_state_dict(checkpoint['model_state_dict'])
        videomae_model = videomae_model.to(train_config.device)
        print(f"Loaded fine-tuned VideoMAE from {finetuned_path}")
        # print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    
    # ================================================================
    # Stage 2: Train AutoGaze
    # ================================================================
    if not args.skip_autogaze:
        print("\n" + "="*60)
        print("STAGE 2: Training AutoGaze")
        print("="*60)
        
        # from train_autogaze import train_autogaze
        # from autogaze import SimpleAutoGaze
        from autogaze_v2 import TrainableAutoGaze as SimpleAutoGaze
        from train_autogaze_v4 import train_autogaze_v2 as train_autogaze
        
        autogaze_model = SimpleAutoGaze(autogaze_config)
        autogaze_model = autogaze_model.to(train_config.device)
        
        # Check for checkpoint to resume from
        autogaze_path = os.path.join(train_config.save_dir, 'autogaze_best.pt')
        start_epoch = 0
        optimizer_state = None
        
        if args.resume_autogaze and os.path.exists(autogaze_path):
            print(f"Resuming AutoGaze from {autogaze_path}")
            checkpoint = torch.load(autogaze_path, map_location=train_config.device, weights_only=False)
            autogaze_model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            if 'optimizer_state_dict' in checkpoint:
                optimizer_state = checkpoint['optimizer_state_dict']
            print(f"  Resuming from epoch {start_epoch}")
            # print(f"  Previous val loss: {checkpoint.get('val_loss', 'N/A')}")
        elif os.path.exists(autogaze_path) and not args.force_retrain:
            print(f"Loading pretrained AutoGaze from {autogaze_path}")
            checkpoint = torch.load(autogaze_path, map_location=train_config.device, weights_only=False)
            autogaze_model.load_state_dict(checkpoint['model_state_dict'])
            # print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        else:
            print("Training AutoGaze from scratch")
        
        # Only train if we haven't finished all epochs
        if start_epoch < train_config.autogaze_epochs:
            if start_epoch > 0:
                print(f"Continuing training from epoch {start_epoch} to {train_config.autogaze_epochs}")
            
            autogaze_model, train_losses, val_losses, train_sparsity = train_autogaze(
                config, train_loader, val_loader, videomae_model,
                model=autogaze_model,  # Pass existing model
                start_epoch=start_epoch,
                optimizer_state=optimizer_state,
            )
            
            # Plot training curves
            from visualize import plot_training_curves
            plot_training_curves(
                train_losses, val_losses,
                save_path=os.path.join(train_config.save_dir, 'autogaze_curves.png'),
            )
        else:
            print(f"AutoGaze already trained for {train_config.autogaze_epochs} epochs, skipping training")
    
    else:
        autogaze_model = None
    
    # ================================================================
    # Stage 3: Comprehensive Evaluation
    # ================================================================
    if not args.skip_eval:
        print("\n" + "="*60)
        print("STAGE 3: Comprehensive Evaluation")
        print("="*60)
        
        from evaluate import evaluate_all_models, print_results
        from visualize import visualize_predictions, visualize_patch_selection
        
        results, all_visualizations = evaluate_all_models(
            config, val_loader, videomae_model, autogaze_model,
        )
        
        print_results(results)
        
        # Save results
        import json
        results_path = os.path.join(train_config.save_dir, 'results.json')
        results_json = {}
        for k, v in results.items():
            if isinstance(v, (np.floating, np.integer)):
                results_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            else:
                results_json[k] = v
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Visualize
        if all_visualizations:
            print("\nGenerating visualizations...")
            vis_data = all_visualizations[0]
            
            # Create GIF for each method
            methods = ['full', 'autogaze', 'random', 'diff', 'center']
            for method in methods:
                if f'pred_{method}' in vis_data:
                    create_input_output_gif(
                        vis_data,
                        save_path=os.path.join(args.save_dir, f'prediction_{method}.gif'),
                        method=method,
                        fps=3,
                    )
            
            # Create comparison GIF (all methods side by side)
            create_comparison_gif(
                vis_data,
                save_path=os.path.join(args.save_dir, 'comparison.gif'),
                fps=3,
            )
            
            # Create patch selection GIF
            if autogaze_model is not None:
                create_patch_selection_gif(
                    vis_data,
                    save_path=os.path.join(args.save_dir, 'patches.gif'),
                    fps=2,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MovingMNIST Video Prediction with AutoGaze')
    
    # Data
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.npy',
                        help='Path to MovingMNIST dataset')
    parser.add_argument('--input_frames', type=int, default=10,
                        help='Number of input frames')
    parser.add_argument('--pred_frames', type=int, default=5,
                        help='Number of frames to predict')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--moving_bg', action='store_true',
                        help='Use moving background dataset')
    
    # Model
    parser.add_argument('--max_patches', type=int, default=16,
                        help='Maximum patches for AutoGaze')
    parser.add_argument('--no_multiscale', action='store_true',
                        help='Disable multi-scale patches')
    
    # Training
    parser.add_argument('--videomae_lr', type=float, default=1e-3,
                        help='Learning rate for VideoMAE')
    parser.add_argument('--videomae_epochs', type=int, default=50,
                        help='Number of VideoMAE epochs')
    parser.add_argument('--autogaze_lr', type=float, default=1e-4,
                        help='Learning rate for AutoGaze')
    parser.add_argument('--autogaze_epochs', type=int, default=30,
                        help='Number of AutoGaze epochs')
    
    # Execution control
    parser.add_argument('--skip_videomae', action='store_true',
                        help='Skip VideoMAE training (load checkpoint)')
    parser.add_argument('--skip_autogaze', action='store_true',
                        help='Skip AutoGaze training (load checkpoint)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining from scratch (ignore checkpoints)')
    parser.add_argument('--resume_videomae', action='store_true',
                        help='Resume VideoMAE training from checkpoint')
    parser.add_argument('--resume_autogaze', action='store_true',
                        help='Resume AutoGaze training from checkpoint')
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")
    
    main(args)