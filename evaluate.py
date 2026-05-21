# evaluate.py - Fixed version

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

def compute_metrics(pred, target):
    """
    Compute MSE, MAE, and SSIM between prediction and target.
    
    pred, target: (B, T, C, H, W) tensors
    Returns dict with scalar metrics averaged over batch and time
    """
    B, T, C, H, W = pred.shape
    
    # MSE and MAE over all dimensions
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    
    # SSIM computed per frame and averaged
    ssim_total = 0.0
    for t in range(T):
        pred_frame = pred[:, t, :, :, :]  # (B, C, H, W)
        target_frame = target[:, t, :, :, :]  # (B, C, H, W)
        ssim_total += compute_ssim(pred_frame, target_frame)
    ssim = ssim_total / T
    
    return {'mse': mse, 'mae': mae, 'ssim': ssim}


def compute_ssim(img1, img2, window_size=11):
    """
    Simplified SSIM computation for 4D tensors (B, C, H, W).
    
    img1, img2: (B, C, H, W) tensors
    Returns scalar SSIM value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian-like kernel
    kernel = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size ** 2)
    # Repeat for each channel
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)  # (C, 1, window, window)
    
    # Compute means using grouped convolution
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def evaluate_videomae_model(model, dataloader, device='cuda'):
    """
    Quick evaluation of a single model.
    Returns metrics and sample predictions.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    count = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            pred_frames = model(inputs)
            
            metrics = compute_metrics(pred_frames, targets)
            
            B = inputs.shape[0]
            total_mse += metrics['mse'] * B
            total_mae += metrics['mae'] * B
            total_ssim += metrics['ssim'] * B
            count += B
            
            # Store first batch for visualization
            if len(all_preds) < 5:
                all_preds.append(pred_frames.cpu())
                all_targets.append(targets.cpu())
    
    return {
        'mse': total_mse / count,
        'mae': total_mae / count,
        'ssim': total_ssim / count,
    }, all_preds, all_targets


def evaluate_all_models(config, val_loader, videomae_model, autogaze_model):
    """
    Evaluate and compare:
    1. Full VideoMAE (all patches) - baseline
    2. AutoGaze + VideoMAE - learned selection
    3. Random patch selection
    4. Heuristic patch selection (difference-based)
    5. Heuristic patch selection (center-based)
    """
    device = config.train_config.device
    
    videomae_model.eval()
    if autogaze_model is not None:
        autogaze_model.eval()
    
    from autogaze import HeuristicPatchSelector

    patch_size = config.videomae_config.patch_size  # 4
    max_patches = config.autogaze_config.max_patches_per_frame  # 64

    random_selector = HeuristicPatchSelector(method='random', num_patches=max_patches, patch_size=patch_size)
    diff_selector = HeuristicPatchSelector(method='difference', num_patches=max_patches, patch_size=patch_size)
    center_selector = HeuristicPatchSelector(method='center', num_patches=max_patches, patch_size=patch_size)
    
    results = defaultdict(list)
    latencies = defaultdict(list)
    
    all_visualizations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Evaluating all models')):
            inputs = batch['input'].to(device)    # (B, T_in, 1, 64, 64)
            targets = batch['target'].to(device)  # (B, T_pred, 1, 64, 64)
            B = inputs.shape[0]
            
            # 1. Full VideoMAE (all patches)
            torch.cuda.synchronize()
            t0 = time.time()
            pred_full = videomae_model(inputs)
            torch.cuda.synchronize()
            latency_full = (time.time() - t0) / B
            
            metrics_full = compute_metrics(pred_full, targets)
            for k, v in metrics_full.items():
                results[f'full_{k}'].append(v)
            latencies['full'].append(latency_full)
            # Full model uses all patches
            results['full_num_patches'].append(videomae_model.num_patches)
            
            # 2. AutoGaze + VideoMAE (only if trained)
            if autogaze_model is not None:
                torch.cuda.synchronize()
                t0 = time.time()
                gaze_output = autogaze_model(inputs)
                patch_mask = gaze_output['patch_mask']
                pred_autogaze = videomae_model.get_sparse_forward(inputs, patch_mask)
                torch.cuda.synchronize()
                latency_autogaze = (time.time() - t0) / B
                
                metrics_autogaze = compute_metrics(pred_autogaze, targets)
                for k, v in metrics_autogaze.items():
                    results[f'autogaze_{k}'].append(v)
                latencies['autogaze'].append(latency_autogaze)
                results['autogaze_num_patches'].append(
                    patch_mask.sum(dim=-1).float().mean().item()
                )
                gaze_data = gaze_output
            else:
                pred_autogaze = pred_full  # fallback for visualization
                gaze_data = None
            
            # 3. Random selection
            torch.cuda.synchronize()
            t0 = time.time()
            rand_mask = random_selector.select_patches(inputs).to(device)
            pred_random = videomae_model.get_sparse_forward(inputs, rand_mask)
            torch.cuda.synchronize()
            latency_random = (time.time() - t0) / B
            
            metrics_random = compute_metrics(pred_random, targets)
            for k, v in metrics_random.items():
                results[f'random_{k}'].append(v)
            latencies['random'].append(latency_random)
            results['random_num_patches'].append(max_patches)
            
            # 4. Difference-based heuristic
            torch.cuda.synchronize()
            t0 = time.time()
            diff_mask = diff_selector.select_patches(inputs).to(device)
            pred_diff = videomae_model.get_sparse_forward(inputs, diff_mask)
            torch.cuda.synchronize()
            latency_diff = (time.time() - t0) / B
            
            metrics_diff = compute_metrics(pred_diff, targets)
            for k, v in metrics_diff.items():
                results[f'diff_{k}'].append(v)
            latencies['diff'].append(latency_diff)
            results['diff_num_patches'].append(max_patches)
            
            # 5. Center heuristic
            torch.cuda.synchronize()
            t0 = time.time()
            center_mask = center_selector.select_patches(inputs).to(device)
            pred_center = videomae_model.get_sparse_forward(inputs, center_mask)
            torch.cuda.synchronize()
            latency_center = (time.time() - t0) / B
            
            metrics_center = compute_metrics(pred_center, targets)
            for k, v in metrics_center.items():
                results[f'center_{k}'].append(v)
            latencies['center'].append(latency_center)
            results['center_num_patches'].append(max_patches)
            
            # Store first batch for visualization
            if len(all_visualizations) == 0:
                all_visualizations.append({
                    'inputs': inputs.cpu(),
                    'targets': targets.cpu(),
                    'pred_full': pred_full.cpu(),
                    'pred_autogaze': pred_autogaze.cpu(),
                    'pred_random': pred_random.cpu(),
                    'pred_diff': pred_diff.cpu(),
                    'pred_center': pred_center.cpu(),
                    'gaze_output': gaze_data,
                })
    
    # Aggregate results
    final_results = {}
    for k, v in results.items():
        final_results[k] = np.mean(v)
    
    for k, v in latencies.items():
        if v:
            final_results[f'latency_{k}'] = np.mean(v) * 1000  # Convert to ms
    
    return final_results, all_visualizations


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    methods = ['full', 'autogaze', 'random', 'diff', 'center']
    method_names = {
        'full': 'Full VideoMAE (all patches)',
        'autogaze': 'AutoGaze + VideoMAE',
        'random': 'Random Selection',
        'diff': 'Frame Difference',
        'center': 'Center Patches',
    }
    
    print(f"\n{'Method':<35} {'MSE':>10} {'MAE':>10} {'SSIM':>10} {'Lat(ms)':>10} {'Patches':>8}")
    print("-"*85)
    
    for method in methods:
        mse = results.get(f'{method}_mse', float('nan'))
        mae = results.get(f'{method}_mae', float('nan'))
        ssim = results.get(f'{method}_ssim', float('nan'))
        latency = results.get(f'latency_{method}', float('nan'))
        patches = results.get(f'{method}_num_patches', float('nan'))
        
        if np.isnan(mse):
            continue  # Skip methods that weren't evaluated
        
        print(f"{method_names.get(method, method):<35} {mse:>10.6f} {mae:>10.6f} {ssim:>10.6f} {latency:>10.2f} {patches:>8.1f}")
    
    print("="*80)
    
    # Key comparisons
    if 'autogaze_mse' in results and 'full_mse' in results:
        mse_increase = (results['autogaze_mse'] - results['full_mse']) / results['full_mse'] * 100
        print(f"\nAutoGaze MSE increase over full: {mse_increase:.1f}%")
    else:
        print("\nAutoGaze not evaluated (train AutoGaze first with --skip_videomae)")
    
    if 'autogaze_num_patches' in results and 'full_num_patches' in results:
        patch_reduction = (1 - results['autogaze_num_patches'] / results['full_num_patches']) * 100
        print(f"AutoGaze patch reduction: {patch_reduction:.1f}%")