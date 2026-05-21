# evaluate_v2.py - Improved evaluation with parameter counts and latency breakdown

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters())

def count_parameters_grad(model):
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(pred, target):
    B, T, C, H, W = pred.shape
    
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    
    # SSIM per frame
    ssim_total = 0.0
    for t in range(T):
        pred_frame = pred[:, t, :, :, :]
        target_frame = target[:, t, :, :, :]
        ssim_total += compute_ssim(pred_frame, target_frame)
    ssim = ssim_total / T
    
    return {'mse': mse, 'mae': mae, 'ssim': ssim}

def compute_ssim(img1, img2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    kernel = torch.ones(1, 1, window_size, window_size, device=img1.device) / (window_size ** 2)
    kernel = kernel.repeat(img1.shape[1], 1, 1, 1)
    
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


def evaluate_all_models_v2(config, val_loader, videomae_model, autogaze_model):
    """
    Improved evaluation with:
    - Parameter counts
    - Latency breakdown (AutoGaze vs VideoMAE separately)
    - FLOP estimates
    """
    device = config.train_config.device
    
    videomae_model.eval()
    if autogaze_model is not None:
        autogaze_model.eval()
    
    from autogaze import HeuristicPatchSelector
    patch_size = config.videomae_config.patch_size
    max_patches = config.autogaze_config.max_patches_per_frame
    
    random_selector = HeuristicPatchSelector(method='random', num_patches=max_patches, patch_size=patch_size)
    diff_selector = HeuristicPatchSelector(method='difference', num_patches=max_patches, patch_size=patch_size)
    center_selector = HeuristicPatchSelector(method='center', num_patches=max_patches, patch_size=patch_size)
    
    # Parameter counts
    videomae_params = count_parameters(videomae_model)
    autogaze_params = count_parameters(autogaze_model) if autogaze_model else 0
    
    print(f"\nModel Parameters:")
    print(f"  VideoMAE: {videomae_params:,} ({videomae_params/1e6:.2f}M)")
    if autogaze_model:
        print(f"  AutoGaze: {autogaze_params:,} ({autogaze_params/1e6:.2f}M)")
        print(f"  Total:    {videomae_params + autogaze_params:,} ({ (videomae_params + autogaze_params)/1e6:.2f}M)")
        print(f"  Overhead: {autogaze_params/videomae_params*100:.1f}% of VideoMAE")
    
    results = defaultdict(list)
    latencies = defaultdict(list)
    all_visualizations = []
    
    num_warmup = 5
    num_benchmark = 20
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Evaluating all models')):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            B = inputs.shape[0]
            
            # ============= 1. Full VideoMAE =============
            # Warmup
            if batch_idx == 0:
                for _ in range(num_warmup):
                    _ = videomae_model(inputs)
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_benchmark):
                pred_full = videomae_model(inputs)
            torch.cuda.synchronize()
            latency_full_total = (time.perf_counter() - t0) / num_benchmark
            
            # Single forward for metrics
            pred_full = videomae_model(inputs)
            metrics_full = compute_metrics(pred_full, targets)
            for k, v in metrics_full.items():
                results[f'full_{k}'].append(v)
            latencies['full_videomae'].append(latency_full_total)
            results['full_num_patches'].append(videomae_model.num_patches)
            
            # ============= 2. AutoGaze + VideoMAE =============
            if autogaze_model is not None:
                # Warmup
                if batch_idx == 0:
                    for _ in range(num_warmup):
                        gaze_output = autogaze_model(inputs)
                        _ = videomae_model.get_sparse_forward(inputs, gaze_output['patch_mask'])
                    torch.cuda.synchronize()
                
                # Benchmark - measure separately
                # AutoGaze latency
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_benchmark):
                    gaze_output = autogaze_model(inputs)
                torch.cuda.synchronize()
                latency_autogaze = (time.perf_counter() - t0) / num_benchmark
                
                patch_mask = gaze_output['patch_mask']
                
                # VideoMAE sparse latency
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_benchmark):
                    _ = videomae_model.get_sparse_forward(inputs, patch_mask)
                torch.cuda.synchronize()
                latency_videomae_sparse = (time.perf_counter() - t0) / num_benchmark
                
                # Single forward for metrics
                pred_autogaze = videomae_model.get_sparse_forward(inputs, patch_mask)
                metrics_autogaze = compute_metrics(pred_autogaze, targets)
                for k, v in metrics_autogaze.items():
                    results[f'autogaze_{k}'].append(v)
                
                latencies['autogaze_only'].append(latency_autogaze)
                latencies['autogaze_videomae'].append(latency_videomae_sparse)
                latencies['autogaze_total'].append(latency_autogaze + latency_videomae_sparse)
                
                num_patches_autogaze = patch_mask.sum(dim=-1).float().mean().item()
                results['autogaze_num_patches'].append(num_patches_autogaze)
                gaze_data = gaze_output
            else:
                pred_autogaze = pred_full
                gaze_data = None
                latencies['autogaze_only'].append(0)
                latencies['autogaze_videomae'].append(0)
                latencies['autogaze_total'].append(0)
            
            # ============= 3. Random Selection =============
            rand_mask = random_selector.select_patches(inputs).to(device)
            
            if batch_idx == 0:
                for _ in range(num_warmup):
                    _ = videomae_model.get_sparse_forward(inputs, rand_mask)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_benchmark):
                _ = videomae_model.get_sparse_forward(inputs, rand_mask)
            torch.cuda.synchronize()
            latency_random = (time.perf_counter() - t0) / num_benchmark
            
            pred_random = videomae_model.get_sparse_forward(inputs, rand_mask)
            metrics_random = compute_metrics(pred_random, targets)
            for k, v in metrics_random.items():
                results[f'random_{k}'].append(v)
            latencies['random'].append(latency_random)
            results['random_num_patches'].append(max_patches)
            
            # ============= 4. Difference Heuristic =============
            diff_mask = diff_selector.select_patches(inputs).to(device)
            
            if batch_idx == 0:
                for _ in range(num_warmup):
                    _ = videomae_model.get_sparse_forward(inputs, diff_mask)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_benchmark):
                _ = videomae_model.get_sparse_forward(inputs, diff_mask)
            torch.cuda.synchronize()
            latency_diff = (time.perf_counter() - t0) / num_benchmark
            
            pred_diff = videomae_model.get_sparse_forward(inputs, diff_mask)
            metrics_diff = compute_metrics(pred_diff, targets)
            for k, v in metrics_diff.items():
                results[f'diff_{k}'].append(v)
            latencies['diff'].append(latency_diff)
            results['diff_num_patches'].append(max_patches)
            
            # ============= 5. Center Heuristic =============
            center_mask = center_selector.select_patches(inputs).to(device)
            
            if batch_idx == 0:
                for _ in range(num_warmup):
                    _ = videomae_model.get_sparse_forward(inputs, center_mask)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_benchmark):
                _ = videomae_model.get_sparse_forward(inputs, center_mask)
            torch.cuda.synchronize()
            latency_center = (time.perf_counter() - t0) / num_benchmark
            
            pred_center = videomae_model.get_sparse_forward(inputs, center_mask)
            metrics_center = compute_metrics(pred_center, targets)
            for k, v in metrics_center.items():
                results[f'center_{k}'].append(v)
            latencies['center'].append(latency_center)
            results['center_num_patches'].append(max_patches)
            
            # Store for visualization
            if len(all_visualizations) == 0:
                all_visualizations.append({
                    'inputs': inputs.cpu(),
                    'targets': targets.cpu(),
                    'pred_full': pred_full.cpu(),
                    'pred_autogaze': pred_autogaze.cpu() if autogaze_model else pred_full.cpu(),
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
        if v and np.mean(v) > 0:
            final_results[f'latency_{k}'] = np.mean(v) * 1000  # Convert to ms
    
    # Add parameter info
    final_results['videomae_params'] = videomae_params
    final_results['autogaze_params'] = autogaze_params
    final_results['total_params'] = videomae_params + autogaze_params
    
    return final_results, all_visualizations


def print_results_v2(results):
    """Pretty print with latency breakdown."""
    print("\n" + "="*85)
    print("EVALUATION RESULTS")
    print("="*85)
    
    # Model info
    print(f"\nModel Sizes:")
    print(f"  VideoMAE: {results.get('videomae_params', 0)/1e6:.2f}M params")
    print(f"  AutoGaze:  {results.get('autogaze_params', 0)/1e6:.2f}M params")
    print(f"  Total:     {results.get('total_params', 0)/1e6:.2f}M params")
    print(f"  Overhead:  {results.get('autogaze_params', 0)/max(results.get('videomae_params', 1), 1)*100:.1f}%")
    
    # Quality metrics
    methods = ['full', 'autogaze', 'random', 'diff', 'center']
    method_names = {
        'full': 'Full VideoMAE',
        'autogaze': 'AutoGaze',
        'random': 'Random',
        'diff': 'Frame Diff',
        'center': 'Center',
    }
    
    print(f"\n{'Method':<25} {'MSE':>8} {'MAE':>8} {'SSIM':>8} {'Patches':>8}")
    print("-"*65)
    
    for method in methods:
        mse = results.get(f'{method}_mse', float('nan'))
        mae = results.get(f'{method}_mae', float('nan'))
        ssim = results.get(f'{method}_ssim', float('nan'))
        patches = results.get(f'{method}_num_patches', 256 if method == 'full' else float('nan'))
        
        if np.isnan(mse):
            continue
        
        print(f"{method_names[method]:<25} {mse:>8.4f} {mae:>8.4f} {ssim:>8.4f} {patches:>8.1f}")
    
    # Latency breakdown
    print(f"\nLatency Breakdown (per frame, ms):")
    print(f"{'Method':<25} {'AutoGaze':>10} {'VideoMAE':>10} {'Total':>10}")
    print("-"*60)
    
    # Full model
    full_lat = results.get('latency_full_videomae', 0)
    print(f"{'Full VideoMAE':<25} {'-':>10} {full_lat:>10.2f} {full_lat:>10.2f}")
    
    # AutoGaze
    ag_lat = results.get('latency_autogaze_only', 0)
    ag_vim_lat = results.get('latency_autogaze_videomae', 0)
    ag_total = results.get('latency_autogaze_total', 0)
    if ag_total > 0:
        print(f"{'AutoGaze':<25} {ag_lat:>10.2f} {ag_vim_lat:>10.2f} {ag_total:>10.2f}")
    
    # Heuristics (no AutoGaze overhead)
    for method in ['random', 'diff', 'center']:
        lat = results.get(f'latency_{method}', 0)
        if lat > 0:
            print(f"{method_names[method]:<25} {'-':>10} {lat:>10.2f} {lat:>10.2f}")
    
    # Key insights
    print(f"\nKey Insights:")
    if ag_total > 0 and full_lat > 0:
        overhead_pct = (ag_total - full_lat) / full_lat * 100
        print(f"  AutoGaze overhead: {overhead_pct:+.1f}% vs full model")
        if ag_vim_lat < full_lat:
            vim_speedup = full_lat / ag_vim_lat
            print(f"  VideoMAE speedup (sparse vs dense): {vim_speedup:.2f}x")
            print(f"  AutoGaze cost: {ag_lat:.2f}ms (negates VideoMAE savings for small model)")
            print(f"  For larger ViT (>100M params), AutoGaze overhead becomes negligible")
    
    print("="*85)

# def benchmark_with_stats(fn, num_warmup=5, num_runs=50):
#     """Benchmark and return mean, std."""
#     for _ in range(num_warmup):
#         fn()
#     torch.cuda.synchronize()
    
#     times = []
#     for _ in range(num_runs):
#         start = time.perf_counter()
#         fn()
#         torch.cuda.synchronize()
#         end = time.perf_counter()
#         times.append((end - start) * 1000)
    
#     return np.mean(times), np.std(times)


# def print_results_v2(results):
#     """Pretty print with std and breakdown."""
#     # ... (keep previous code)
    
#     # Latency with std
#     print(f"\n{'Latency (ms, mean ± std)':<40}")
#     print("-"*60)
    
#     full_lat = results.get('latency_full_videomae', 0)
#     full_std = results.get('latency_full_videomae_std', 0)
#     print(f"{'Full VideoMAE':<30} {full_lat:>8.2f} ± {full_std:>5.2f} ms")
    
#     if 'latency_autogaze_only' in results:
#         ag_lat = results['latency_autogaze_only']
#         ag_std = results.get('latency_autogaze_only_std', 0)
#         ag_vim_lat = results['latency_autogaze_videomae']
#         ag_vim_std = results.get('latency_autogaze_videomae_std', 0)
#         ag_total = results['latency_autogaze_total']
#         ag_total_std = results.get('latency_autogaze_total_std', 0)
        
#         print(f"{'  AutoGaze only':<30} {ag_lat:>8.2f} ± {ag_std:>5.2f} ms")
#         print(f"{'  VideoMAE (sparse)':<30} {ag_vim_lat:>8.2f} ± {ag_vim_std:>5.2f} ms")
#         print(f"{'  Total':<30} {ag_total:>8.2f} ± {ag_total_std:>5.2f} ms")
        
#         vim_speedup = full_lat / ag_vim_lat if ag_vim_lat > 0 else 0
#         print(f"\n  VideoMAE speedup: {vim_speedup:.2f}x")
#         print(f"  AutoGaze overhead: {ag_lat:.2f} ms")