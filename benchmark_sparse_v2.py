# benchmark_sparse_v2.py - Better benchmark with std and breakdown

import torch
import numpy as np
import time
from einops import rearrange

def benchmark_with_stats(fn, num_warmup=10, num_runs=100, name=""):
    """Benchmark a function and return mean + std."""
    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg = np.mean(times)
    std = np.std(times)
    
    return avg, std


def full_benchmark():
    """Comprehensive latency benchmark."""
    device = 'cuda'
    B, T, C, H, W = 1, 10, 1, 64, 64
    patch_size = 4
    grid_size = H // patch_size
    num_patches = grid_size * grid_size  # 256
    
    # Load model
    from videomae_v2 import MemorySafeVideoMAE
    from config import VideoMAEConfig
    
    config = VideoMAEConfig()
    model = MemorySafeVideoMAE(config)
    checkpoint = torch.load('./checkpoints_v2/videomae_best.pt', 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Try to load AutoGaze
    autogaze = None
    try:
        from autogaze_v2 import TrainableAutoGaze
        from config import AutoGazeConfig
        ag_config = AutoGazeConfig(img_size=64, patch_size=4, max_patches_per_frame=64)
        autogaze = TrainableAutoGaze(ag_config)
        checkpoint = torch.load('./checkpoints_v2/autogaze_best.pt', 
                               map_location=device, weights_only=False)
        autogaze.load_state_dict(checkpoint['model_state_dict'])
        autogaze = autogaze.to(device)
        autogaze.eval()
    except:
        print("No AutoGaze checkpoint found, skipping AutoGaze benchmarks")
    
    # Create input
    x = torch.randn(B, T, C, H, W, device=device)
    
    # Create masks at different sparsity levels
    sparsity_levels = {
        "100%": 256,
        "50%": 128,
        "25%": 64,
        "12.5%": 32,
        "6.25%": 16,
    }
    
    masks = {}
    for name, k in sparsity_levels.items():
        mask = torch.zeros(B, T, num_patches, device=device)
        for b in range(B):
            for t in range(T):
                indices = torch.randperm(num_patches, device=device)[:max(1, k)]
                mask[b, t, indices] = 1.0
        masks[name] = mask
    
    # ============= VIDEO MAE ONLY =============
    print("\n" + "="*80)
    print("VIDEO MAE (ViT) LATENCY BREAKDOWN")
    print("="*80)
    
    # Full forward
    avg_full, std_full = benchmark_with_stats(
        lambda: model(x),
        name="Full VideoMAE (256 patches)"
    )
    print(f"{'Full VideoMAE (256 patches)':<40} {avg_full:>8.2f} ± {std_full:>6.2f} ms")
    
    # Sparse forward (zero-masked) at different sparsity levels
    print(f"\n{'Sparse (zero-masked) - processes all patches':<40}")
    print("-"*60)
    for name, k in sparsity_levels.items():
        if k == 256:
            continue
        mask = masks[name]
        avg, std = benchmark_with_stats(
            lambda m=mask: model.get_sparse_forward(x, m),
            name=f"  {name} ({k} patches)"
        )
        speedup = avg_full / avg
        print(f"  {name} ({k:3d} patches){' ':>20} {avg:>8.2f} ± {std:>6.2f} ms  ({speedup:.2f}x)")
    
    # ============= AUTOGAZE + VIDEO MAE =============
    if autogaze is not None:
        print("\n" + "="*80)
        print("AUTOGAZE + VIDEO MAE LATENCY BREAKDOWN")
        print("="*80)
        
        # AutoGaze only
        avg_ag, std_ag = benchmark_with_stats(
            lambda: autogaze(x),
            name="AutoGaze only"
        )
        print(f"{'AutoGaze only (patch selection)':<40} {avg_ag:>8.2f} ± {std_ag:>6.2f} ms")
        print(f"  Params: {sum(p.numel() for p in autogaze.parameters())/1e6:.2f}M")
        
        # VideoMAE with AutoGaze-selected patches
        with torch.no_grad():
            gaze_output = autogaze(x)
        ag_mask = gaze_output['patch_mask']
        num_ag_patches = ag_mask.sum().item()
        
        avg_ag_vim, std_ag_vim = benchmark_with_stats(
            lambda m=ag_mask: model.get_sparse_forward(x, m),
            name="VideoMAE with AutoGaze patches"
        )
        print(f"{'VideoMAE with AutoGaze patches':<40} {avg_ag_vim:>8.2f} ± {std_ag_vim:>6.2f} ms")
        
        # Total
        total_ag = avg_ag + avg_ag_vim
        print(f"{'Total (AutoGaze + VideoMAE)':<40} {total_ag:>8.2f} ± {std_ag + std_ag_vim:>6.2f} ms")
        print(f"  AutoGaze selected: {num_ag_patches} patches ({num_ag_patches/num_patches*100:.1f}%)")
        
        # Speedup analysis
        print(f"\n{'Speedup Analysis':<40}")
        print("-"*60)
        vim_speedup = avg_full / avg_ag_vim
        print(f"  VideoMAE speedup (sparse vs full): {vim_speedup:.2f}x")
        overhead = total_ag - avg_full
        print(f"  AutoGaze overhead: {overhead:+.2f} ms ({overhead/avg_full*100:+.1f}%)")
        
        # Theoretical for larger ViT
        print(f"\n{'Theoretical for Larger ViTs':<40}")
        print("-"*60)
        vit_sizes = [5, 20, 86, 300]  # Millions of params
        overhead_ratio = avg_ag / avg_full  # AutoGaze overhead relative to full model
        
        for vit_size in vit_sizes:
            # Assume latency scales roughly with params
            scaled_full = avg_full * (vit_size / 4)  # Our model is ~4M
            scaled_ag = avg_ag  # AutoGaze stays constant
            scaled_vim_sparse = avg_ag_vim * (vit_size / 4) * (num_ag_patches / num_patches)
            scaled_total = scaled_ag + scaled_vim_sparse
            scaled_speedup = scaled_full / scaled_total
            print(f"  ViT-{vit_size}M: {scaled_speedup:.1f}x speedup (full: {scaled_full:.1f}ms -> sparse: {scaled_total:.1f}ms)")
    
    # ============= SUMMARY TABLE =============
    print("\n" + "="*80)
    print("LATENCY SUMMARY (mean ± std, ms)")
    print("="*80)
    print(f"{'Method':<35} {'Latency (ms)':>20} {'Speedup':>10}")
    print("-"*70)
    print(f"{'Full VideoMAE':<35} {avg_full:>13.2f} ± {std_full:>5.2f}  {'1.00x':>10}")
    
    if autogaze is not None:
        print(f"{'AutoGaze only':<35} {avg_ag:>13.2f} ± {std_ag:>5.2f}  {'---':>10}")
        print(f"{'VideoMAE (sparse)':<35} {avg_ag_vim:>13.2f} ± {std_ag_vim:>5.2f}  {vim_speedup:>9.2f}x")
        print(f"{'Total (AG + ViM)':<35} {total_ag:>13.2f} ± {std_ag+std_ag_vim:>5.2f}  {avg_full/total_ag:>9.2f}x")


if __name__ == '__main__':
    full_benchmark()