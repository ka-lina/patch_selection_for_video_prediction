# benchmark_sparse.py - Fixed to match current model

import torch
import torch.nn as nn
import time
import numpy as np
from einops import rearrange

class EfficientSparseForward:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, x, patch_mask):
        return self.sparse_forward_efficient(x, patch_mask)
    
    def sparse_forward_efficient(self, x, patch_mask):
        """Process only selected patches through spatial encoder."""
        B, T, C, H, W = x.shape
        
        # Get patch embeddings for ALL patches
        patches = self.model.patch_embed(x)
        patches = patches + self.model.pos_embed[:, :, :, :]
        patches = patches + self.model.temp_pos_embed[:, :T, :, :]
        
        # Reshape for per-frame processing
        patches = rearrange(patches, 'b t n d -> (b t) n d')
        mask_flat = rearrange(patch_mask, 'b t n -> (b t) n')
        
        BT, N, D = patches.shape
        outputs = torch.zeros(BT, N, D, device=patches.device)
        
        # Process only selected patches through spatial encoder
        for i in range(BT):
            selected_idx = mask_flat[i].bool()
            if selected_idx.sum() > 0:
                selected_patches = patches[i, selected_idx, :]  # (k, D)
                
                # Run through spatial transformer blocks (only on selected!)
                for block in self.model.spatial_blocks:
                    selected_patches = block(selected_patches)
                
                # Scatter back to full grid
                outputs[i, selected_idx, :] = selected_patches
        
        # Now follow the EXACT same flow as model.encode() but with pre-computed outputs
        # The original encode() does:
        #   1. spatial encoding -> spatial_pooled
        #   2. temporal encoding -> temporal_pooled
        #   3. fuse spatial_2d + temporal_modulation
        # We've done step 1 efficiently. Now do steps 2-3.
        
        # Temporal pooling
        x_reshaped = rearrange(outputs, '(b t) n d -> b t n d', b=B, t=T)
        x_spatial_pooled = x_reshaped.mean(dim=2)  # (B, T, D)
        
        # Spatial 2D features for decoder
        x_spatial_2d = rearrange(outputs, '(b t) (g1 g2) d -> (b t) d g1 g2',
                                b=B, t=T, g1=self.model.grid_size, g2=self.model.grid_size)
        
        # Temporal encoding
        temporal_tokens = x_spatial_pooled.unsqueeze(2)  # (B, T, 1, D)
        temporal_tokens = temporal_tokens + self.model.temporal_token[:, :, :, :]
        temporal_tokens = temporal_tokens + self.model.temp_pos_embed[:, :T, :, :]
        temporal_tokens = rearrange(temporal_tokens, 'b t n d -> (b t) n d')
        
        for block in self.model.temporal_blocks:
            temporal_tokens = block(temporal_tokens)
        
        temporal_tokens = rearrange(temporal_tokens, '(b t) n d -> b t n d', b=B, t=T)
        temporal_pooled = temporal_tokens.mean(dim=1).squeeze(1)  # (B, D)
        
        # Fuse: spatial + temporal modulation
        temporal_modulation = temporal_pooled.unsqueeze(-1).unsqueeze(-1)
        temporal_modulation = temporal_modulation.expand(-1, -1, self.model.grid_size, self.model.grid_size)
        
        x_spatial_2d = rearrange(x_spatial_2d, '(b t) d g1 g2 -> b t d g1 g2', b=B, t=T)
        x_spatial_2d = x_spatial_2d.mean(dim=1)  # (B, D, grid, grid)
        
        fused = x_spatial_2d + temporal_modulation  # (B, D, grid, grid)
        
        # Use the model's own decode method
        # decode() expects (encoded, T_enc, num_pred_frames) where encoded is (B, T*N, D)
        # But fused is already (B, D, grid, grid) - the decoded spatial features
        # So we need to use the decoder convolutions directly
        
        # The decode method does:
        #   1. Reshape encoded to (B, D, grid, grid)
        #   2. decoder_up1, decoder_up2, final_conv
        # We already have the (B, D, grid, grid) features!
        
        x = self.model.decoder_up1(fused)   # (B, 64, 32, 32)
        x = self.model.decoder_up2(x)        # (B, 32, 64, 64)
        x = self.model.final_conv(x)         # (B, T_pred*C, 64, 64)
        x = x.reshape(B, 5, self.model.in_channels, 
                      self.model.img_size, self.model.img_size)
        
        return x

def benchmark_sparse_forward():
    """
    Compare latency of:
    1. Full forward pass (all 256 patches)
    2. Current sparse forward (mask * zero, processes all 256)
    3. Efficient sparse forward (only processes selected k patches)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, C, H, W = 1, 10, 1, 64, 64
    patch_size = 4
    grid_size = H // patch_size
    num_patches = grid_size * grid_size  # 256
    
    # Load your model - use the current config
    from videomae_v2 import MemorySafeVideoMAE

    from config import DataConfig, VideoMAEConfig, AutoGazeConfig, TrainConfig
    
    # Create configs
    data_config = DataConfig(
        dataset_path="mnist_test_seq.npy",
        input_frames=10,
        pred_frames=5,
        batch_size=16,
        use_moving_background=False,
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
    
    # Load checkpoint with strict=False to handle potential mismatches
    checkpoint = torch.load('./checkpoints_v2/videomae_best.pt', 
                           map_location=device, weights_only=False)
    
    # Try loading with strict=True first
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("Loaded checkpoint with exact match")
    except RuntimeError as e:
        print(f"Could not load exactly: {e}")
        print("Trying with strict=False...")
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(B, T, C, H, W, device=device)
    
    # Create different sparsity masks
    sparsity_levels = [1.0, 0.5, 0.25, 0.125, 0.0625]
    masks = {}
    for ratio in sparsity_levels:
        k = max(1, int(num_patches * ratio))
        mask = torch.zeros(B, T, num_patches, device=device)
        for b in range(B):
            for t in range(T):
                indices = torch.randperm(num_patches, device=device)[:k]
                mask[b, t, indices] = 1.0
        masks[ratio] = mask
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            _ = model.get_sparse_forward(x, masks[0.25])
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark function
    def benchmark(fn, *args, num_runs=50, name=""):
        if device == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = fn(*args)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"{name:<50} {avg_time:>8.2f} ms (±{std_time:.2f})")
        return avg_time
    
    print("\n" + "="*70)
    print("LATENCY BENCHMARK (50 runs each, single input)")
    print("="*70)
    print(f"{'Method':<50} {'Time':>10}")
    print("-"*70)
    
    results = {}
    
    # 1. Full forward pass (baseline)
    results['full'] = benchmark(
        lambda: model(x), 
        num_runs=50, 
        name="Full forward (all 256 patches)"
    )
    
    # 2. Current sparse forward
    for ratio in sparsity_levels[1:]:
        k = max(1, int(num_patches * ratio))
        mask = masks[ratio]
        results[f'current_sparse_{ratio}'] = benchmark(
            lambda m=mask: model.get_sparse_forward(x, m),
            num_runs=50,
            name=f"Current sparse ({k} patches, masked-zero)"
        )
    
    # Efficient sparse forward (actually skips patches)
    efficient_model = EfficientSparseForward(model)
    for ratio in sparsity_levels[1:]:
        k = max(1, int(num_patches * ratio))
        mask = masks[ratio]
        results[f'efficient_{ratio}'] = benchmark(
            lambda m=mask: efficient_model(x, m),
            name=f"Efficient sparse ({k:3d} patches, real skip)"
        )
    
    # Print speedup summary
    print("\n" + "="*70)
    print("SPEEDUP vs FULL FORWARD")
    print("="*70)
    print(f"{'Patches':<15} {'Current Method':>15} {'Efficient Method':>18}")
    print("-"*55)
    
    full_time = results['full']
    for ratio in sparsity_levels[1:]:
        k = max(1, int(num_patches * ratio))
        current_time = results.get(f'current_sparse_{ratio}', full_time)
        efficient_time = results.get(f'efficient_sparse_{ratio}', full_time)
        
        current_speedup = full_time / current_time if current_time > 0 else 0
        efficient_speedup = full_time / efficient_time if efficient_time > 0 else 0
        
        print(f"{k:>4} ({ratio:.0%})        {current_speedup:>10.2f}x         {efficient_speedup:>14.2f}x")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
The 'Current Method' processes all 256 patches but multiplies
unselected ones by zero. This gives NO speedup (may be slower
due to mask overhead).

The 'Efficient Method' actually skips unselected patches in the
spatial encoder. This should show speedup proportional to the
number of patches actually processed.

NOTE: The Python for-loop over batch items adds overhead.
A production implementation would batch variable-length sequences
for even greater speedup (as in the original AutoGaze paper).
""")


if __name__ == '__main__':
    benchmark_sparse_forward()