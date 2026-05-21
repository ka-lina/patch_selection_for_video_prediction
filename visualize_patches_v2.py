# visualize_patches_v2.py - Improved patch visualization

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os

def create_patch_selection_gif(vis_data, save_path, fps=2, num_examples=3):
    """
    GIF showing input frames with selected patches highlighted in red.
    One GIF per example, plays through all input frames.
    """
    if 'gaze_output' not in vis_data or vis_data['gaze_output'] is None:
        print("No AutoGaze data available for patch visualization")
        return
    
    gaze_output = vis_data['gaze_output']
    patch_mask = gaze_output['patch_mask']  # (B, T, N)
    inputs = vis_data['inputs']  # (B, T, 1, H, W)
    
    B = min(inputs.shape[0], num_examples)
    T = inputs.shape[1]
    grid_size = int(patch_mask.shape[-1] ** 0.5)
    patch_size_px = inputs.shape[-1] // grid_size
    
    for b in range(B):
        frames = []
        
        for t in range(T):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Convert grayscale to RGB for colored overlay
            frame = inputs[b, t, 0].cpu().numpy()
            rgb = np.stack([frame, frame, frame], axis=-1)
            
            # Highlight selected patches
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if patch_mask[b, t, idx] > 0.5:
                        y1, y2 = i * patch_size_px, (i+1) * patch_size_px
                        x1, x2 = j * patch_size_px, (j+1) * patch_size_px
                        # Red tint on selected patches
                        rgb[y1:y2, x1:x2, 0] = np.maximum(rgb[y1:y2, x1:x2, 0], 0.8)
                        rgb[y1:y2, x1:x2, 1:] *= 0.3
            
            ax.imshow(rgb)
            num_selected = patch_mask[b, t].sum().item()
            total = patch_mask.shape[-1]
            ax.set_title(f'Frame {t+1}/{T} | {num_selected:.0f}/{total} patches ({num_selected/total:.1%})', 
                        fontsize=12)
            ax.axis('off')
            
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
        print(f"Saved patch selection GIF to {gif_path}")


def create_patch_selection_static(vis_data, save_path, num_examples=3, max_frames=5):
    """
    Static grid showing input frames and selected patches for multiple examples.
    Rows: examples, Columns: frames (up to max_frames).
    Selected patches outlined in red, unselected patches dimmed.
    """
    if 'gaze_output' not in vis_data or vis_data['gaze_output'] is None:
        print("No AutoGaze data available for patch visualization")
        return
    
    gaze_output = vis_data['gaze_output']
    patch_mask = gaze_output['patch_mask']  # (B, T, N)
    inputs = vis_data['inputs']  # (B, T, 1, H, W)
    
    B = min(inputs.shape[0], num_examples)
    T = min(inputs.shape[1], max_frames)
    grid_size = int(patch_mask.shape[-1] ** 0.5)
    patch_size_px = inputs.shape[-1] // grid_size
    
    fig, axes = plt.subplots(B, T + 1, figsize=(3 * (T + 1), 3.5 * B))
    
    if B == 1:
        axes = axes[np.newaxis, :]
    
    for b in range(B):
        # First column: show the last predicted frame + ground truth
        if 'pred_autogaze' in vis_data and 'targets' in vis_data:
            pred = vis_data['pred_autogaze'][b, -1, 0].cpu().numpy()
            gt = vis_data['targets'][b, -1, 0].cpu().numpy()
            mse = np.mean((pred - gt) ** 2)
            
            # Overlay prediction with ground truth difference
            diff = np.abs(pred - gt)
            overlay = np.stack([
                pred,  # R: prediction
                np.maximum(pred - diff * 2, 0),  # G: reduced where error
                np.maximum(pred - diff * 2, 0),  # B: reduced where error
            ], axis=-1)
            axes[b, 0].imshow(overlay)
            axes[b, 0].set_title(f'Prediction\nMSE: {mse:.4f}', fontsize=9)
        else:
            axes[b, 0].axis('off')
            axes[b, 0].set_title('')
        axes[b, 0].axis('off')
        
        for t in range(T):
            frame = inputs[b, t, 0].cpu().numpy()
            
            # Create RGB image
            rgb = np.stack([frame, frame, frame], axis=-1)
            
            # Draw patch grid overlay
            ax = axes[b, t + 1]
            ax.imshow(rgb)
            
            # Draw grid lines
            for i in range(grid_size + 1):
                pos = i * patch_size_px
                ax.axhline(y=pos, color='white', linewidth=0.5, alpha=0.3)
                ax.axvline(x=pos, color='white', linewidth=0.5, alpha=0.3)
            
            # Highlight selected patches
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if patch_mask[b, t, idx] > 0.5:
                        rect = Rectangle(
                            (j * patch_size_px, i * patch_size_px),
                            patch_size_px, patch_size_px,
                            linewidth=2, edgecolor='red', 
                            facecolor='red', alpha=0.3
                        )
                        ax.add_patch(rect)
            
            num_selected = patch_mask[b, t].sum().item()
            total = patch_mask.shape[-1]
            ax.set_title(f'Frame {t+1}\n{num_selected:.0f}/{total} ({num_selected/total:.1%})', 
                        fontsize=9)
            ax.axis('off')
    
    plt.suptitle('AutoGaze Patch Selection - Input Frames with Selected Patches (Red)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved static patch visualization to {save_path}")
    plt.show()


def create_patch_summary_chart(vis_data, save_path):
    """
    Bar chart showing average patches selected per frame across the dataset.
    Also shows comparison with heuristic methods.
    """
    if 'gaze_output' not in vis_data or vis_data['gaze_output'] is None:
        print("No AutoGaze data for summary chart")
        return
    
    gaze_output = vis_data['gaze_output']
    patch_mask = gaze_output['patch_mask']  # (B, T, N)
    
    T = patch_mask.shape[1]
    total_patches = patch_mask.shape[-1]
    
    # Average patches per frame
    avg_patches = patch_mask.sum(dim=-1).float().mean(dim=0).cpu().numpy()  # (T,)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: patches per frame
    colors = plt.cm.Reds(0.3 + 0.7 * avg_patches / total_patches)
    ax1.bar(range(T), avg_patches, color=colors, edgecolor='darkred', linewidth=1)
    ax1.axhline(y=total_patches, color='gray', linestyle='--', alpha=0.5, label=f'Max ({total_patches})')
    ax1.axhline(y=16, color='blue', linestyle='--', alpha=0.5, label='Heuristic (16)')
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Average Patches Selected', fontsize=12)
    ax1.set_title('AutoGaze: Patches Selected per Frame', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, total_patches * 1.1)
    
    # Add percentage labels on bars
    for i, v in enumerate(avg_patches):
        pct = v / total_patches * 100
        ax1.text(i, v + 2, f'{pct:.1f}%', ha='center', fontsize=8)
    
    # Pie chart: overall sparsity
    avg_total = avg_patches.mean()
    selected_pct = avg_total / total_patches * 100
    unselected_pct = 100 - selected_pct
    
    wedges, texts, autotexts = ax2.pie(
        [selected_pct, unselected_pct], 
        labels=['Selected', 'Unselected'],
        colors=['#ff4444', '#dddddd'],
        autopct='%1.1f%%',
        explode=(0.05, 0),
        startangle=90,
    )
    ax2.set_title(f'Overall Sparsity\n{avg_total:.1f} / {total_patches} patches per frame', 
                 fontsize=13, fontweight='bold')
    
    plt.suptitle('AutoGaze Patch Selection Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved summary chart to {save_path}")
    plt.show()


def create_all_visualizations(vis_data, save_dir, num_examples=3):
    """
    Create all patch visualizations at once.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. GIF animation
    print("\nCreating patch selection GIF...")
    create_patch_selection_gif(
        vis_data,
        save_path=os.path.join(save_dir, 'patches.gif'),
        fps=2,
        num_examples=num_examples,
    )
    
    # 2. Static grid
    print("Creating static patch grid...")
    create_patch_selection_static(
        vis_data,
        save_path=os.path.join(save_dir, 'patches_static.png'),
        num_examples=num_examples,
        max_frames=5,
    )
    
    # 3. Summary chart
    print("Creating summary chart...")
    create_patch_summary_chart(
        vis_data,
        save_path=os.path.join(save_dir, 'patches_summary.png'),
    )