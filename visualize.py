# visualize.py - Updated with GIF support

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
import os

def create_prediction_gif(vis_data, save_path, method='full', fps=2, num_examples=3):
    """
    Create a GIF comparing ground truth vs prediction frames over time.
    
    Args:
        vis_data: dict with keys 'inputs', 'targets', 'pred_full', 'pred_autogaze', etc.
        save_path: path to save the GIF
        method: which prediction method to visualize ('full', 'autogaze', 'random', 'diff', 'center')
        fps: frames per second in the GIF
        num_examples: number of examples to show
    """
    pred_key = f'pred_{method}'
    if pred_key not in vis_data:
        print(f"No predictions for method '{method}', skipping GIF")
        return
    
    inputs = vis_data['inputs']  # (B, T_in, 1, H, W)
    targets = vis_data['targets']  # (B, T_pred, 1, H, W)
    predictions = vis_data[pred_key]  # (B, T_pred, 1, H, W)
    
    B = min(inputs.shape[0], num_examples)
    T_in = inputs.shape[1]
    T_pred = targets.shape[1]
    
    method_names = {
        'full': 'Full VideoMAE',
        'autogaze': 'AutoGaze',
        'random': 'Random',
        'diff': 'Frame Diff',
        'center': 'Center',
    }
    method_name = method_names.get(method, method)
    
    # Create figure with rows for examples, columns for time steps
    total_frames = T_in + T_pred  # Show all input + prediction frames
    fig, axes = plt.subplots(B, total_frames, figsize=(3 * total_frames, 3 * B))
    
    if B == 1:
        axes = axes[np.newaxis, :]
    
    # Create frames for the animation
    frames_data = []
    
    for t in range(total_frames):
        # Clear axes for this frame
        for b in range(B):
            for col in range(total_frames):
                axes[b, col].clear()
                axes[b, col].axis('off')
        
        for b in range(B):
            # Show input frames (always visible)
            for t_in in range(min(T_in, t + 1)):
                frame = inputs[b, t_in, 0].numpy()
                axes[b, t_in].imshow(frame, cmap='gray', vmin=0, vmax=1)
                axes[b, t_in].set_title(f'Input t={t_in}', fontsize=8)
                axes[b, t_in].axis('off')
            
            # Show prediction/ground truth up to current time
            for t_pred in range(min(T_pred, t - T_in + 1)):
                col = T_in + t_pred
                
                if col < total_frames:
                    # Show ground truth
                    gt_frame = targets[b, t_pred, 0].numpy()
                    pred_frame = predictions[b, t_pred, 0].numpy()
                    
                    # Create side-by-side or overlaid comparison
                    # Here we show prediction with ground truth color-coded difference
                    diff = np.abs(pred_frame - gt_frame)
                    
                    # Overlay: prediction as grayscale, error as red
                    rgb_frame = np.stack([
                        pred_frame,  # R: prediction
                        pred_frame - diff * 0.5,  # G: reduced where error
                        pred_frame - diff * 0.5,  # B: reduced where error
                    ], axis=-1)
                    rgb_frame = np.clip(rgb_frame, 0, 1)
                    
                    axes[b, col].imshow(rgb_frame)
                    
                    mse = np.mean((pred_frame - gt_frame) ** 2)
                    axes[b, col].set_title(f'Pred t={t_pred}\nMSE={mse:.4f}', fontsize=8)
                    axes[b, col].axis('off')
        
        # Convert current figure state to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:,:,1:] # argb to rgb
        frames_data.append(image)
    
    plt.close(fig)
    
    # Save as GIF
    gif_path = save_path.replace('.png', f'_{method}.gif') if save_path.endswith('.png') else save_path
    frames_pil = [Image.fromarray(frame) for frame in frames_data]
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(1000 / fps),  # ms per frame
        loop=0,  # 0 = infinite loop
    )
    print(f"Saved prediction GIF to {gif_path}")


def create_comparison_gif(vis_data, save_path, fps=2, num_examples=3):
    """
    Create a GIF showing all prediction methods side by side for comparison.
    Only shows the prediction frames (not input frames).
    """
    methods = ['full', 'autogaze', 'random', 'diff', 'center']
    method_names = ['Full', 'AutoGaze', 'Random', 'Diff', 'Center']
    available = [(m, n) for m, n in zip(methods, method_names) 
                 if f'pred_{m}' in vis_data]
    
    if not available:
        print("No predictions available")
        return
    
    targets = vis_data['targets']
    B = min(targets.shape[0], num_examples)
    T_pred = targets.shape[1]
    
    # Create figure: rows for examples, columns for methods + ground truth
    n_cols = len(available) + 1  # +1 for ground truth
    fig, axes = plt.subplots(B, n_cols, figsize=(3 * n_cols, 3 * B))
    
    if B == 1:
        axes = axes[np.newaxis, :]
    
    frames_data = []
    
    for t_pred in range(T_pred):
        # Clear
        for b in range(B):
            for col in range(n_cols):
                axes[b, col].clear()
                axes[b, col].axis('off')
        
        for b in range(B):
            # Ground truth (first column)
            gt_frame = targets[b, t_pred, 0].numpy()
            axes[b, 0].imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
            axes[b, 0].set_title(f'Ground Truth t={t_pred}', fontsize=8)
            axes[b, 0].axis('off')
            
            # Each method
            for col_idx, (method, name) in enumerate(available):
                pred_frame = vis_data[f'pred_{method}'][b, t_pred, 0].numpy()
                mse = np.mean((pred_frame - gt_frame) ** 2)
                
                axes[b, col_idx + 1].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
                axes[b, col_idx + 1].set_title(f'{name}\nMSE={mse:.4f}', fontsize=8)
                axes[b, col_idx + 1].axis('off')
        
        # Add overall title
        fig.suptitle(f'Frame {t_pred + 1}/{T_pred} Predictions', fontsize=12)
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:,:,1:] # argb to rgb
        frames_data.append(image)
    
    plt.close(fig)
    
    gif_path = save_path.replace('.png', '_comparison.gif') if save_path.endswith('.png') else save_path
    frames_pil = [Image.fromarray(frame) for frame in frames_data]
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved comparison GIF to {gif_path}")


def create_patch_selection_gif(vis_data, save_path, fps=2, num_examples=3):
    """
    Create a GIF showing AutoGaze patch selection over time.
    Shows input frames with selected patches highlighted in red.
    """
    if 'gaze_output' not in vis_data or vis_data['gaze_output'] is None:
        print("No AutoGaze data available for patch visualization")
        return
    
    gaze_output = vis_data['gaze_output']
    patch_mask = gaze_output['patch_mask']  # (B, T_in, N)
    inputs = vis_data['inputs']  # (B, T_in, 1, H, W)
    
    B = min(inputs.shape[0], num_examples)
    T_in = inputs.shape[1]
    grid_size = int(patch_mask.shape[-1] ** 0.5)
    patch_size = inputs.shape[-1] // grid_size
    
    fig, axes = plt.subplots(B, min(T_in, 5), figsize=(3 * min(T_in, 5), 3 * B))
    if B == 1:
        axes = axes[np.newaxis, :]
    if min(T_in, 5) == 1:
        axes = axes[:, np.newaxis]
    
    frames_data = []
    display_frames = min(T_in, 5)
    
    for t in range(display_frames):
        for b in range(B):
            axes[b, t].clear()
            axes[b, t].axis('off')
        
        for b in range(B):
            frame = inputs[b, t, 0].numpy()
            
            # Convert grayscale to RGB for colored overlays
            rgb_frame = np.stack([frame, frame, frame], axis=-1)
            
            # Overlay selected patches in red
            for i in range(grid_size):
                for j in range(grid_size):
                    patch_idx = i * grid_size + j
                    if patch_mask[b, t, patch_idx] > 0.5:
                        y_start = i * patch_size
                        x_start = j * patch_size
                        rgb_frame[y_start:y_start+patch_size, x_start:x_start+patch_size, 0] = \
                            np.maximum(rgb_frame[y_start:y_start+patch_size, x_start:x_start+patch_size, 0], 0.8)
                        rgb_frame[y_start:y_start+patch_size, x_start:x_start+patch_size, 1:] *= 0.3
            
            axes[b, t].imshow(rgb_frame)
            
            num_selected = patch_mask[b, t].sum().item()
            axes[b, t].set_title(f'Frame {t}\n{num_selected:.0f}/{patch_mask.shape[-1]:.0f} patches', fontsize=8)
            axes[b, t].axis('off')
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:,:,1:] # argb to rgb
        frames_data.append(image)
    
    plt.close(fig)
    
    gif_path = save_path.replace('.png', '_patches.gif') if save_path.endswith('.png') else save_path
    frames_pil = [Image.fromarray(frame) for frame in frames_data]
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved patch selection GIF to {gif_path}")


def create_input_output_gif(vis_data, save_path, method='full', fps=3, num_examples=3):
    """
    Create a single GIF per example showing:
    - All 10 input frames
    - 5 ground truth frames
    - 5 predicted frames
    All in sequence for smooth animation.
    """
    pred_key = f'pred_{method}'
    if pred_key not in vis_data:
        return
    
    inputs = vis_data['inputs']
    targets = vis_data['targets']
    predictions = vis_data[pred_key]
    
    B = min(inputs.shape[0], num_examples)
    T_in = inputs.shape[1]
    T_pred = targets.shape[1]
    
    method_names = {
        'full': 'Full VideoMAE',
        'autogaze': 'AutoGaze',
        'random': 'Random',
        'diff': 'Frame Diff',
        'center': 'Center',
    }
    method_name = method_names.get(method, method)
    
    for b in range(B):
        frames = []
        
        # Input frames
        for t in range(T_in):
            frame = inputs[b, t, 0].numpy()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Input Frame {t+1}/{T_in}', fontsize=10)
            ax.axis('off')
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(image))
            plt.close(fig)
        
        # Ground truth + Prediction side by side
        for t in range(T_pred):
            gt_frame = targets[b, t, 0].numpy()
            pred_frame = predictions[b, t, 0].numpy()
            mse = np.mean((pred_frame - gt_frame) ** 2)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            ax1.imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
            ax1.set_title(f'Ground Truth t={t+1}/{T_pred}', fontsize=10)
            ax1.axis('off')
            
            ax2.imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f'{method_name}\nMSE={mse:.4f}', fontsize=10)
            ax2.axis('off')
            
            fig.suptitle(f'Example {b+1} - Frame {T_in + t + 1}', fontsize=12)
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(image))
            plt.close(fig)
        
        # Save GIF for this example
        gif_path = save_path.replace('.gif', f'_example{b+1}_{method}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved example {b+1} GIF to {gif_path}")


# Keep the original static visualization functions for backward compatibility
def visualize_predictions(vis_data, save_path=None, num_examples=3):
    """Static visualization - kept for quick previews."""
    B = min(vis_data['inputs'].shape[0], num_examples)
    T_in = vis_data['inputs'].shape[1]
    T_pred = vis_data['targets'].shape[1]
    
    methods = ['full', 'autogaze', 'random', 'diff', 'center']
    method_names = ['Full', 'AutoGaze', 'Random', 'Diff', 'Center']
    available_methods = [(m, n) for m, n in zip(methods, method_names) 
                         if f'pred_{m}' in vis_data]
    
    fig, axes = plt.subplots(B, len(available_methods) + 1, 
                              figsize=(4 * (len(available_methods) + 1), 3 * B))
    if B == 1:
        axes = axes[np.newaxis, :]
    
    for b in range(B):
        last_input = vis_data['inputs'][b, -1, 0].numpy()
        axes[b, 0].imshow(last_input, cmap='gray')
        axes[b, 0].set_title('Last Input')
        axes[b, 0].axis('off')
        
        gt_frame = vis_data['targets'][b, -1, 0].numpy()
        
        for col_idx, (method, name) in enumerate(available_methods):
            pred_frame = vis_data[f'pred_{method}'][b, -1, 0].numpy()
            axes[b, col_idx + 1].imshow(pred_frame, cmap='gray')
            mse = np.mean((pred_frame - gt_frame) ** 2)
            axes[b, col_idx + 1].set_title(f'{name}\nMSE: {mse:.4f}')
            axes[b, col_idx + 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved static visualization to {save_path}")
    plt.show()


def visualize_patch_selection(vis_data, save_path=None, num_examples=3):
    """Static patch visualization - kept for quick previews."""
    if 'gaze_output' not in vis_data or vis_data['gaze_output'] is None:
        return
    
    gaze_output = vis_data['gaze_output']
    patch_mask = gaze_output['patch_mask']
    inputs = vis_data['inputs']
    
    B = min(inputs.shape[0], num_examples)
    T_in = inputs.shape[1]
    grid_size = int(patch_mask.shape[-1] ** 0.5)
    patch_size = inputs.shape[-1] // grid_size
    
    fig, axes = plt.subplots(B, min(T_in, 5), figsize=(3 * min(T_in, 5), 3 * B))
    if B == 1:
        axes = axes[np.newaxis, :]
    
    for b in range(B):
        for t in range(min(T_in, 5)):
            frame = inputs[b, t, 0].numpy()
            axes[b, t].imshow(frame, cmap='gray')
            
            for i in range(grid_size):
                for j in range(grid_size):
                    patch_idx = i * grid_size + j
                    if patch_mask[b, t, patch_idx] > 0.5:
                        rect = Rectangle(
                            (j * patch_size, i * patch_size),
                            patch_size, patch_size,
                            linewidth=1, edgecolor='red', facecolor='none', alpha=0.7
                        )
                        axes[b, t].add_patch(rect)
            
            axes[b, t].set_title(f'Frame {t}\n{patch_mask[b,t].sum():.0f} patches')
            axes[b, t].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved static patch visualization to {save_path}")
    plt.show()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_losses, label='Train')
    ax2.plot(epochs, val_losses, label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Curves (Log)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    plt.show()

def create_input_output_gif(vis_data, save_path, method='full', fps=3, num_examples=3):
    """
    Create a single GIF per example showing:
    - Input frames (top row, all 10 shown as they play)
    - Ground truth (bottom-left) vs Prediction (bottom-right) side by side
    
    The animation plays through all frames sequentially:
    First the 10 input frames, then the 5 prediction frames with GT comparison.
    """
    pred_key = f'pred_{method}'
    if pred_key not in vis_data:
        print(f"No predictions for '{method}', skipping GIF")
        return
    
    inputs = vis_data['inputs']      # (B, T_in, 1, H, W)
    targets = vis_data['targets']    # (B, T_pred, 1, H, W)
    predictions = vis_data[pred_key] # (B, T_pred, 1, H, W)
    
    B = min(inputs.shape[0], num_examples)
    T_in = inputs.shape[1]
    T_pred = targets.shape[1]
    total_frames = T_in + T_pred
    
    method_names = {
        'full': 'Full VideoMAE',
        'autogaze': 'AutoGaze',
        'random': 'Random',
        'diff': 'Frame Diff',
        'center': 'Center',
    }
    method_name = method_names.get(method, method)
    
    for b in range(B):
        frames = []
        
        for t in range(total_frames):
            fig, axes = plt.subplots(2, 2, figsize=(8, 6), 
                                     gridspec_kw={'height_ratios': [1, 1]})
            
            # Determine which phase we're in
            if t < T_in:
                # Input phase: show current input frame
                input_frame = inputs[b, t, 0].numpy()
                
                # Top-left: Current input frame
                axes[0, 0].imshow(input_frame, cmap='gray', vmin=0, vmax=1)
                axes[0, 0].set_title(f'Input Frame {t+1}/{T_in}', fontsize=11, fontweight='bold')
                axes[0, 0].axis('off')
                
                # Top-right: Show previous input frames as reference
                # (or just show a placeholder for the first frame)
                axes[0, 1].imshow(input_frame, cmap='gray', vmin=0, vmax=1, alpha=0.3)
                axes[0, 1].set_title('(Awaiting predictions...)', fontsize=10, color='gray')
                axes[0, 1].axis('off')
                
                # Bottom row: Placeholder for predictions
                axes[1, 0].text(0.5, 0.5, 'Ground Truth\n(starts at t=10)', 
                               ha='center', va='center', fontsize=12, color='gray',
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Ground Truth', fontsize=10)
                axes[1, 0].axis('off')
                
                axes[1, 1].text(0.5, 0.5, f'{method_name}\n(starts at t=10)', 
                               ha='center', va='center', fontsize=12, color='gray',
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Prediction', fontsize=10)
                axes[1, 1].axis('off')
                
                phase_text = f'INPUT PHASE - Frame {t+1}/{T_in}'
                
            else:
                # Prediction phase: show last input + GT vs Prediction
                t_pred = t - T_in  # 0-indexed prediction frame
                
                # Top-left: Last input frame (persistent reference)
                last_input = inputs[b, -1, 0].numpy()
                axes[0, 0].imshow(last_input, cmap='gray', vmin=0, vmax=1)
                axes[0, 0].set_title(f'Last Input Frame (t={T_in-1})', fontsize=10, color='blue')
                axes[0, 0].axis('off')
                
                # Top-right: Show which prediction frame we're on
                # Display a timeline or just the current prediction frame number
                if t_pred > 0:
                    prev_pred = predictions[b, t_pred - 1, 0].numpy()
                    axes[0, 1].imshow(prev_pred, cmap='gray', vmin=0, vmax=1, alpha=0.5)
                    axes[0, 1].set_title(f'Previous Prediction (t={t_pred-1})', fontsize=10, color='gray')
                else:
                    axes[0, 1].imshow(last_input, cmap='gray', vmin=0, vmax=1, alpha=0.3)
                    axes[0, 1].set_title('Start of predictions', fontsize=10, color='gray')
                axes[0, 1].axis('off')
                
                # Bottom-left: Ground truth for current prediction frame
                gt_frame = targets[b, t_pred, 0].numpy()
                axes[1, 0].imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title(f'Ground Truth t={t_pred+1}/{T_pred}', fontsize=11, 
                                    fontweight='bold', color='green')
                axes[1, 0].axis('off')
                
                # Bottom-right: Prediction for current frame
                pred_frame = predictions[b, t_pred, 0].numpy()
                mse = np.mean((pred_frame - gt_frame) ** 2)
                axes[1, 1].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title(f'{method_name} t={t_pred+1}/{T_pred}\nMSE: {mse:.4f}', 
                                    fontsize=11, fontweight='bold', color='red')
                axes[1, 1].axis('off')
                
                phase_text = f'PREDICTION PHASE - Frame {t_pred+1}/{T_pred}'
            
            fig.suptitle(f'Example {b+1} | {method_name} | {phase_text}', 
                        fontsize=13, fontweight='bold')
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(image))
            plt.close(fig)
        
        # Save GIF
        gif_path = save_path.replace('.gif', f'_example{b+1}_{method}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved {method_name} example {b+1} GIF to {gif_path}")


def create_prediction_only_gif(vis_data, save_path, method='full', fps=2, num_examples=3):
    """
    Create a GIF showing ONLY the prediction frames (5 frames).
    Ground truth on left, prediction on right, side by side.
    Much cleaner and focused comparison.
    """
    pred_key = f'pred_{method}'
    if pred_key not in vis_data:
        return
    
    targets = vis_data['targets']    # (B, T_pred, 1, H, W)
    predictions = vis_data[pred_key] # (B, T_pred, 1, H, W)
    
    B = min(targets.shape[0], num_examples)
    T_pred = targets.shape[1]
    
    method_names = {
        'full': 'Full VideoMAE',
        'autogaze': 'AutoGaze',
        'random': 'Random',
        'diff': 'Frame Diff',
        'center': 'Center',
    }
    method_name = method_names.get(method, method)
    
    for b in range(B):
        frames = []
        
        for t in range(T_pred):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # Left: Ground Truth
            gt_frame = targets[b, t, 0].numpy()
            ax1.imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
            ax1.set_title(f'Ground Truth\nFrame {t+1}/{T_pred}', fontsize=12, 
                         fontweight='bold', color='green')
            ax1.axis('off')
            
            # Right: Prediction
            pred_frame = predictions[b, t, 0].numpy()
            mse = np.mean((pred_frame - gt_frame) ** 2)
            ax2.imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
            ax2.set_title(f'{method_name}\nFrame {t+1}/{T_pred} | MSE: {mse:.4f}', 
                         fontsize=12, fontweight='bold', color='red')
            ax2.axis('off')
            
            fig.suptitle(f'Example {b+1} | {method_name} | Prediction Frame {t+1}/{T_pred}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(image))
            plt.close(fig)
        
        gif_path = save_path.replace('.gif', f'_example{b+1}_{method}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved {method_name} example {b+1} GIF to {gif_path}")


def create_multi_method_comparison_gif(vis_data, save_path, fps=2, num_examples=3):
    """
    Create a GIF showing ALL methods compared at once.
    
    Layout for each prediction frame:
    Row 1: Ground Truth | Full | AutoGaze
    Row 2: Random | Diff | Center
    """
    methods = ['full', 'autogaze', 'random', 'diff', 'center']
    method_names = ['Full VideoMAE', 'AutoGaze', 'Random', 'Frame Diff', 'Center']
    
    available = [(m, n) for m, n in zip(methods, method_names) 
                 if f'pred_{m}' in vis_data]
    
    if len(available) < 2:
        print("Need at least 2 methods for comparison GIF")
        return
    
    targets = vis_data['targets']
    B = min(targets.shape[0], num_examples)
    T_pred = targets.shape[1]
    
    for b in range(B):
        frames = []
        
        for t in range(T_pred):
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            gt_frame = targets[b, t, 0].numpy()
            
            # Layout mapping
            layout = {
                0: (0, 0),  # Ground Truth - top left
            }
            
            # Assign methods to grid positions
            for idx, (method, name) in enumerate(available):
                if idx == 0:
                    layout[method] = (0, 1)  # First method - top middle
                elif idx == 1:
                    layout[method] = (0, 2)  # Second method - top right
                elif idx == 2:
                    layout[method] = (1, 0)  # Third method - bottom left
                elif idx == 3:
                    layout[method] = (1, 1)  # Fourth method - bottom middle
                elif idx == 4:
                    layout[method] = (1, 2)  # Fifth method - bottom right
            
            # Ground Truth
            ax = axes[0, 0]
            ax.imshow(gt_frame, cmap='gray', vmin=0, vmax=1)
            ax.set_title('Ground Truth', fontsize=11, fontweight='bold', color='green')
            ax.axis('off')
            
            # Each method
            for method, name in available:
                row, col = layout[method]
                ax = axes[row, col]
                pred_frame = vis_data[f'pred_{method}'][b, t, 0].numpy()
                mse = np.mean((pred_frame - gt_frame) ** 2)
                
                ax.imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{name}\nMSE: {mse:.4f}', fontsize=10)
                ax.axis('off')
            
            # Hide unused subplots
            used_positions = set()
            used_positions.add((0, 0))
            for method, _ in available:
                used_positions.add(layout[method])
            
            for row in range(2):
                for col in range(3):
                    if (row, col) not in used_positions:
                        axes[row, col].axis('off')
            
            fig.suptitle(f'Example {b+1} | Prediction Frame {t+1}/{T_pred} | All Methods Comparison', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:,:,1:] # argb to rgb
            frames.append(Image.fromarray(image))
            plt.close(fig)
        
        gif_path = save_path.replace('.gif', f'_example{b+1}_all_methods.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved all-methods comparison example {b+1} GIF to {gif_path}")