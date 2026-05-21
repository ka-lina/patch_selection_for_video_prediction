# eval_only.py - Run evaluation without training

import torch
import os
import argparse
import numpy as np
import random
import json

from visualize import create_input_output_gif, create_comparison_gif

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(args):
    # Import configs
    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    
    # Setup
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load configs
    data_config = DataConfig(
        dataset_path=args.data_path,
        input_frames=args.input_frames,
        pred_frames=args.pred_frames,
        batch_size=args.batch_size,
    )
    
    # videomae_config = VideoMAEConfig(
    #     img_size=data_config.img_size,
    #     embed_dim=256,
    #     depth=6,
    #     num_heads=8,
    #     decoder_depth=3,
    #     decoder_num_heads=4,
    # )
    from config import VideoMAEConfig

    videomae_config = VideoMAEConfig(
        img_size=64,
        patch_size=8,        # Original used 8×8 patches
        in_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=4.0,       # Original used 4.0
    )

    # autogaze_config = AutoGazeConfig(
    #     img_size=data_config.img_size,
    #     max_patches_per_frame=args.max_patches,
    # )
    autogaze_config = AutoGazeConfig(
        img_size=64,
        patch_size=8,        # Match original VideoMAE
        max_patches_per_frame=16,  # 16 out of 64 patches = 25%
    )
    
    train_config = TrainConfig(
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
    )
    
    class Config:
        pass
    config = Config()
    config.data_config = data_config
    config.videomae_config = videomae_config
    config.autogaze_config = autogaze_config
    config.train_config = train_config
    
    # Create dataloaders
    from dataset import create_dataloaders
    _, val_loader = create_dataloaders(config)
    print(f"Validation batches: {len(val_loader)}")
    
    # Load VideoMAE
    from videomae import PredictiveVideoMAE
    # from videomae_v2 import MemorySafeVideoMAE as PredictiveVideoMAE
    videomae_path = args.videomae_checkpoint or os.path.join(args.save_dir, 'videomae_best.pt') #'videomae_best.pt')
    
    if not os.path.exists(videomae_path):
        raise FileNotFoundError(f"VideoMAE checkpoint not found at {videomae_path}")
    
    print(f"Loading VideoMAE from {videomae_path}")
    checkpoint = torch.load(videomae_path, map_location=device, weights_only=False)
    videomae_model = PredictiveVideoMAE(videomae_config)
    videomae_model.load_state_dict(checkpoint['model_state_dict'])
    videomae_model = videomae_model.to(device)
    videomae_model.eval()
    
    # Print checkpoint info
    print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.6f}" if 'train_loss' in checkpoint else "")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}" if 'val_loss' in checkpoint else "")
    
    # Load AutoGaze (if exists)
    autogaze_model = None
    autogaze_path = args.autogaze_checkpoint or os.path.join(args.save_dir, 'autogaze_recurrent_best.pt')
    
    if os.path.exists(autogaze_path):
        print(f"Loading AutoGaze from {autogaze_path}")
        # from autogaze import SimpleAutoGaze
        checkpoint_ag = torch.load(autogaze_path, map_location=device, weights_only=False)
        from autogaze_v3_recurrent import RecurrentAutoGaze
        from config import AutoGazeConfig
        autogaze_config = AutoGazeConfig(img_size=64, patch_size=4, max_patches_per_frame=64)
        model = RecurrentAutoGaze(config.autogaze_config)
        autogaze_model.load_state_dict(checkpoint_ag['model_state_dict'])
        autogaze_model = autogaze_model.to(device)
        autogaze_model.eval()
        
        
        print(f"  Trained for {checkpoint_ag.get('epoch', 'unknown')} epochs")
        print(f"  Train loss: {checkpoint_ag.get('train_loss', 'N/A'):.6f}" if 'train_loss' in checkpoint_ag else "")
        print(f"  Val loss: {checkpoint_ag.get('val_loss', 'N/A'):.6f}" if 'val_loss' in checkpoint_ag else "")
    else:
        print(f"No AutoGaze checkpoint found at {autogaze_path}")
        print("Will evaluate full VideoMAE and heuristic baselines only.")
    
    # Run evaluation
    # from evaluate import evaluate_all_models, print_results
    from evaluate_v2 import evaluate_all_models_v2 as evaluate_all_models
    from evaluate_v2 import print_results_v2 as print_results
    from visualize import visualize_predictions, visualize_patch_selection, create_prediction_only_gif, create_multi_method_comparison_gif, create_patch_selection_gif
    
    print("\nRunning evaluation...")
    results, all_visualizations = evaluate_all_models(
        config, val_loader, videomae_model, autogaze_model
    )
    
    # Print results
    print_results(results)
    
    # Save results
    results_path = os.path.join(args.save_dir, 'results.json')
    # Convert numpy values for JSON serialization
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
        
        # 1. Per-method prediction GIFs (GT vs Prediction side by side)
        print("\nCreating per-method prediction GIFs...")
        methods = ['full', 'autogaze', 'random', 'diff', 'center']
        for method in methods:
            if f'pred_{method}' in vis_data:
                create_prediction_only_gif(
                    vis_data,
                    save_path=os.path.join(args.save_dir, f'prediction_{method}.gif'),
                    method=method,
                    fps=2,
                    num_examples=3,
                )
        
        # 2. All-methods comparison GIF (all methods in one view)
        print("\nCreating all-methods comparison GIF...")
        create_multi_method_comparison_gif(
            vis_data,
            save_path=os.path.join(args.save_dir, 'all_methods_comparison.gif'),
            fps=2,
            num_examples=3,
        )
        
        # # 3. AutoGaze patch selection GIF
        # if autogaze_model is not None:
        #     print("\nCreating patch selection GIF...")
        #     create_patch_selection_gif(
        #         vis_data,
        #         save_path=os.path.join(args.save_dir, 'patch_selection.gif'),
        #         fps=2,
        #         num_examples=3,
        #     )

        from visualize_patches_v2 import create_all_visualizations
    
        create_all_visualizations(
            all_visualizations[0],
            save_dir=args.save_dir,
            num_examples=3,
        )
        
        print(f"\nAll GIFs saved to {args.save_dir}/")
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    parser.add_argument('--data_path', type=str, default='mnist_test_seq.npy')
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--pred_frames', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_patches', type=int, default=16)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory with saved checkpoints')
    parser.add_argument('--videomae_checkpoint', type=str, default=None,
                        help='Path to VideoMAE checkpoint (default: save_dir/videomae_best.pt)')
    parser.add_argument('--autogaze_checkpoint', type=str, default=None,
                        help='Path to AutoGaze checkpoint (default: save_dir/autogaze_best.pt)')
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)