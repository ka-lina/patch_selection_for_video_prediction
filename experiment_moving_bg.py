# experiment_moving_bg.py - With GIF visualization

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
import os

def create_patch_gif(inputs, mask, save_path, fps=3, title=""):
    """
    Create a GIF showing input frames with selected patches highlighted.
    inputs: (B, T, C, H, W)
    mask: (B, T, N) - binary patch selection mask
    """
    B = min(inputs.shape[0], 3)  # Show up to 3 examples
    T = inputs.shape[1]
    grid_size = int(mask.shape[-1] ** 0.5)
    patch_size_px = inputs.shape[-1] // grid_size
    
    for b in range(B):
        frames = []
        
        for t in range(T):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Convert grayscale to RGB for colored overlay
            frame = inputs[b, t, 0].cpu().numpy()
            rgb = np.stack([frame, frame, frame], axis=-1)
            
            # Highlight selected patches in red
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if mask[b, t, idx] > 0.5:
                        y1, y2 = i * patch_size_px, (i+1) * patch_size_px
                        x1, x2 = j * patch_size_px, (j+1) * patch_size_px
                        # Red tint on selected patches
                        rgb[y1:y2, x1:x2, 0] = np.maximum(rgb[y1:y2, x1:x2, 0], 0.8)
                        rgb[y1:y2, x1:x2, 1:] *= 0.3
            
            ax.imshow(rgb)
            num_selected = mask[b, t].sum().item()
            total = mask.shape[-1]
            ax.set_title(f'Frame {t+1}/{T} | {num_selected:.0f}/{total} patches ({num_selected/total:.1%})', 
                        fontsize=12)
            ax.axis('off')
            
            if t == 0:
                ax.text(10, 10, title, fontsize=10, color='white', 
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            plt.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(img))
            plt.close(fig)
        
        # Save GIF
        gif_path = save_path.replace('.gif', f'_example{b+1}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved GIF to {gif_path}")


def create_comparison_gif(inputs, mask_before, mask_after, save_path, fps=3):
    """
    Side-by-side GIF comparing zero-shot vs fine-tuned patch selection.
    Shows the same frame with before/after patch selection.
    """
    b = 0  # First example
    T = inputs.shape[1]
    grid_size = int(mask_before.shape[-1] ** 0.5)
    patch_size_px = inputs.shape[-1] // grid_size
    
    frames = []
    
    for t in range(T):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        frame = inputs[b, t, 0].cpu().numpy()
        
        for ax, mask, title_str in [
            (ax1, mask_before, 'Zero-Shot\n(selects everywhere)'),
            (ax2, mask_after, 'Fine-Tuned\n(focuses on digits)')
        ]:
            rgb = np.stack([frame, frame, frame], axis=-1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if mask[b, t, idx] > 0.5:
                        y1, y2 = i * patch_size_px, (i+1) * patch_size_px
                        x1, x2 = j * patch_size_px, (j+1) * patch_size_px
                        rgb[y1:y2, x1:x2, 0] = np.maximum(rgb[y1:y2, x1:x2, 0], 0.8)
                        rgb[y1:y2, x1:x2, 1:] *= 0.3
            
            ax.imshow(rgb)
            n = mask[b, t].sum().item()
            total = mask.shape[-1]
            ax.set_title(f'{title_str}\n{n:.0f}/{total} patches ({n/total:.1%})', fontsize=11)
            ax.axis('off')
        
        fig.suptitle(f'Frame {t+1}/{T} - AutoGaze Adaptation to Moving Background', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:,:,1:] # argb to rgb
        frames.append(Image.fromarray(img))
        plt.close(fig)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved comparison GIF to {save_path}")


def create_prediction_gif(inputs, targets, pred_before, pred_after, save_path, fps=2):
    """
    Show prediction results: ground truth vs zero-shot vs fine-tuned.
    """
    b = 0
    T_pred = targets.shape[1]
    
    frames = []
    
    for t in range(T_pred):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        gt = targets[b, t, 0].cpu().numpy()
        ax1.imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Ground Truth\nFrame {t+1}', fontsize=12)
        ax1.axis('off')
        
        # Zero-shot prediction
        pred_b = pred_before[b, t, 0].cpu().numpy()
        mse_b = np.mean((pred_b - gt) ** 2)
        ax2.imshow(pred_b, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f'Zero-Shot Prediction\nMSE: {mse_b:.4f}', fontsize=12)
        ax2.axis('off')
        
        # Fine-tuned prediction
        pred_f = pred_after[b, t, 0].cpu().numpy()
        mse_f = np.mean((pred_f - gt) ** 2)
        ax3.imshow(pred_f, cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Fine-Tuned Prediction\nMSE: {mse_f:.4f}', fontsize=12)
        ax3.axis('off')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:,:,1:] # argb to rgb
        frames.append(Image.fromarray(img))
        plt.close(fig)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved prediction GIF to {save_path}")


def run_moving_background_experiment():
    """Test AutoGaze on moving background MovingMNIST."""
    device = 'cuda'
    save_dir = './checkpoints_v2/moving_bg_experiment'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load models
    print("Loading models...")
    from videomae_v2 import MemorySafeVideoMAE
    from autogaze_v2 import TrainableAutoGaze
    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    
    # Create configs
    data_config = DataConfig(
        dataset_path="mnist_test_seq.npy",
        input_frames=10,
        pred_frames=5,
        batch_size=16,
        use_moving_background=False,
    )

    vim_config = VideoMAEConfig(
        img_size=data_config.img_size,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_depth=3,
        decoder_num_heads=4,
    )
    videomae = MemorySafeVideoMAE(vim_config).to(device)
    ckpt = torch.load('./checkpoints_v3/videomae_finetuned.pt', map_location=device, weights_only=False)
    videomae.load_state_dict(ckpt['model_state_dict'])
    videomae.eval()
    
    ag_config = AutoGazeConfig(img_size=64, patch_size=4, max_patches_per_frame=64)
    autogaze = TrainableAutoGaze(ag_config).to(device)
    ckpt = torch.load('./checkpoints_v3/autogaze_best.pt', map_location=device, weights_only=False)
    autogaze.load_state_dict(ckpt['model_state_dict'])
    autogaze.eval()
    
    # 2. Create moving background dataset
    print("Creating moving background dataset...")
    from dataset import MovingMNISTWithBackground, DataLoader
    
    dataset = MovingMNISTWithBackground(
        data_path='mnist_test_seq.npy',
        input_frames=10, pred_frames=5,
        train=True, num_train_videos=200,
        img_size=64, background_speed=2,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    batch = next(iter(loader))
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    # 3. Zero-shot evaluation
    print("\n" + "="*60)
    print("ZERO-SHOT: AutoGaze on Moving Background")
    print("="*60)
    
    with torch.no_grad():
        gaze_out = autogaze(inputs)
        mask_before = gaze_out['patch_mask']
        num_patches_before = mask_before.sum(dim=-1).float().mean().item()
        
        pred_before = videomae.get_sparse_forward(inputs, mask_before)
        mse_before = torch.nn.functional.mse_loss(pred_before, targets).item()
        
        pred_full = videomae(inputs)
        mse_full = torch.nn.functional.mse_loss(pred_full, targets).item()
    
    print(f"AutoGaze patches: {num_patches_before:.1f}/256 ({num_patches_before/256*100:.1f}%)")
    print(f"AutoGaze MSE: {mse_before:.4f}")
    print(f"Full model MSE: {mse_full:.4f}")
    
    # 4. Visualize zero-shot
    print("\nCreating zero-shot visualizations...")
    create_patch_gif(
        inputs, mask_before,
        save_path=os.path.join(save_dir, 'patches_zeroshot.gif'),
        fps=3,
        title='Zero-Shot: AutoGaze selects MANY patches'
    )
    
    # 5. Fine-tune AutoGaze
    print("\n" + "="*60)
    print("FINE-TUNING AutoGaze on Moving Background")
    print("="*60)
    
    autogaze.train()
    optimizer = torch.optim.AdamW(autogaze.parameters(), lr=1e-5)
    
    for epoch in range(15):
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/15')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            gaze_out = autogaze(inputs)
            mask = gaze_out['patch_mask']
            
            pred = videomae.get_sparse_forward(inputs, mask)
            pred_loss = torch.nn.functional.mse_loss(pred, targets)
            
            sparsity = mask.sum(dim=-1).float().mean() / 256
            loss = pred_loss + 0.05 * sparsity
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'patches': f'{mask.sum(dim=-1).float().mean().item():.1f}'})
        
        # Quick eval
        autogaze.eval()
        with torch.no_grad():
            gaze_out = autogaze(inputs)
            mask_eval = gaze_out['patch_mask']
            n = mask_eval.sum(dim=-1).float().mean().item()
            pred = videomae.get_sparse_forward(inputs, mask_eval)
            m = torch.nn.functional.mse_loss(pred, targets).item()
        autogaze.train()
        
        print(f"  Patches: {n:.1f}, MSE: {m:.4f}")
    
    # 6. Evaluate fine-tuned
    print("\n" + "="*60)
    print("AFTER FINE-TUNING")
    print("="*60)
    
    autogaze.eval()
    with torch.no_grad():
        gaze_out = autogaze(inputs)
        mask_after = gaze_out['patch_mask']
        num_patches_after = mask_after.sum(dim=-1).float().mean().item()
        
        pred_after = videomae.get_sparse_forward(inputs, mask_after)
        mse_after = torch.nn.functional.mse_loss(pred_after, targets).item()
    
    print(f"AutoGaze patches: {num_patches_after:.1f}/256 ({num_patches_after/256*100:.1f}%)")
    print(f"AutoGaze MSE: {mse_after:.4f}")
    print(f"Reduction: {num_patches_before:.1f} → {num_patches_after:.1f} patches")
    
    # 7. Create comparison visualizations
    print("\nCreating comparison visualizations...")
    
    # Patch selection comparison GIF
    create_comparison_gif(
        inputs, mask_before, mask_after,
        save_path=os.path.join(save_dir, 'patches_comparison.gif'),
        fps=3,
    )
    
    # Prediction comparison GIF
    create_prediction_gif(
        inputs, targets, pred_before, pred_after,
        save_path=os.path.join(save_dir, 'predictions_comparison.gif'),
        fps=2,
    )
    
    # Fine-tuned patch GIF
    create_patch_gif(
        inputs, mask_after,
        save_path=os.path.join(save_dir, 'patches_finetuned.gif'),
        fps=3,
        title='Fine-Tuned: AutoGaze focuses on digits'
    )
    
    # 8. Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'':<25} {'Patches':>10} {'MSE':>10}")
    print("-"*50)
    print(f"{'Full model':<25} {256:>10.0f} {mse_full:>10.4f}")
    print(f"{'AutoGaze (zero-shot)':<25} {num_patches_before:>10.1f} {mse_before:>10.4f}")
    print(f"{'AutoGaze (fine-tuned)':<25} {num_patches_after:>10.1f} {mse_after:>10.4f}")
    print(f"\nAll results saved to {save_dir}/")


if __name__ == '__main__':
    run_moving_background_experiment()