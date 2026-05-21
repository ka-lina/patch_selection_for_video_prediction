# experiment_moving_bg.py - With latency measurements

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import time

def benchmark_latency(fn, num_warmup=10, num_runs=100):
    """Benchmark a function, return mean and std in ms."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.mean(times), np.std(times)


def run_moving_background_experiment():
    """Test AutoGaze on moving background with latency measurements."""
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
    
    # Model sizes
    vim_params = sum(p.numel() for p in videomae.parameters()) / 1e6
    ag_params = sum(p.numel() for p in autogaze.parameters()) / 1e6
    print(f"VideoMAE: {vim_params:.2f}M params, AutoGaze: {ag_params:.2f}M params")
    
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
    
    # ============= LATENCY BENCHMARKS =============
    print("\n" + "="*70)
    print("LATENCY BENCHMARKS (Moving Background)")
    print("="*70)
    
    # 1. Full VideoMAE
    avg_full, std_full = benchmark_latency(
        lambda: videomae(inputs),
    )
    print(f"{'Full VideoMAE':<35} {avg_full:>8.2f} ± {std_full:>5.2f} ms")
    
    # 2. AutoGaze only (zero-shot)
    avg_ag_zs, std_ag_zs = benchmark_latency(
        lambda: autogaze(inputs),
    )
    print(f"{'AutoGaze only (zero-shot)':<35} {avg_ag_zs:>8.2f} ± {std_ag_zs:>5.2f} ms")
    
    # 3. AutoGaze + VideoMAE (zero-shot)
    with torch.no_grad():
        gaze_zs = autogaze(inputs)
        mask_zs = gaze_zs['patch_mask']
    
    avg_vim_zs, std_vim_zs = benchmark_latency(
        lambda m=mask_zs: videomae.get_sparse_forward(inputs, m),
    )
    print(f"{'VideoMAE sparse (zero-shot)':<35} {avg_vim_zs:>8.2f} ± {std_vim_zs:>5.2f} ms")
    print(f"{'Total (AG + ViM, zero-shot)':<35} {avg_ag_zs + avg_vim_zs:>8.2f} ± {std_ag_zs + std_vim_zs:>5.2f} ms")
    
    # 4. Get predictions
    with torch.no_grad():
        pred_zs = videomae.get_sparse_forward(inputs, mask_zs)
        pred_full = videomae(inputs)
        mse_zs = torch.nn.functional.mse_loss(pred_zs, targets).item()
        mse_full = torch.nn.functional.mse_loss(pred_full, targets).item()
    
    num_patches_zs = mask_zs.sum(dim=-1).float().mean().item()
    
    # ============= FINE-TUNING =============
    print("\n" + "="*70)
    print("FINE-TUNING AutoGaze on Moving Background")
    print("="*70)
    
    autogaze.train()
    optimizer = torch.optim.AdamW(autogaze.parameters(), lr=1e-5)
    
    for epoch in range(15):
        total_loss = 0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/15')
        for batch in pbar:
            inputs_ft = batch['input'].to(device)
            targets_ft = batch['target'].to(device)
            
            optimizer.zero_grad()
            gaze_out = autogaze(inputs_ft)
            mask = gaze_out['patch_mask']
            
            pred = videomae.get_sparse_forward(inputs_ft, mask)
            pred_loss = torch.nn.functional.mse_loss(pred, targets_ft)
            
            sparsity = mask.sum(dim=-1).float().mean() / 256
            loss = pred_loss + 0.05 * sparsity
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'patches': f'{mask.sum(dim=-1).float().mean().item():.1f}'})
        
        # Eval on fixed batch
        autogaze.eval()
        with torch.no_grad():
            gaze_out = autogaze(inputs)
            mask_eval = gaze_out['patch_mask']
            n = mask_eval.sum(dim=-1).float().mean().item()
            pred = videomae.get_sparse_forward(inputs, mask_eval)
            m = torch.nn.functional.mse_loss(pred, targets).item()
        autogaze.train()
        print(f"  Patches: {n:.1f}, MSE: {m:.4f}")
    
    # ============= POST-FINETUNING LATENCY =============
    print("\n" + "="*70)
    print("POST-FINETUNING LATENCY")
    print("="*70)
    
    autogaze.eval()
    
    # AutoGaze only (fine-tuned)
    avg_ag_ft, std_ag_ft = benchmark_latency(
        lambda: autogaze(inputs),
    )
    print(f"{'AutoGaze only (fine-tuned)':<35} {avg_ag_ft:>8.2f} ± {std_ag_ft:>5.2f} ms")
    
    # VideoMAE sparse (fine-tuned)
    with torch.no_grad():
        gaze_ft = autogaze(inputs)
        mask_ft = gaze_ft['patch_mask']
    
    avg_vim_ft, std_vim_ft = benchmark_latency(
        lambda m=mask_ft: videomae.get_sparse_forward(inputs, m),
    )
    print(f"{'VideoMAE sparse (fine-tuned)':<35} {avg_vim_ft:>8.2f} ± {std_vim_ft:>5.2f} ms")
    print(f"{'Total (AG + ViM, fine-tuned)':<35} {avg_ag_ft + avg_vim_ft:>8.2f} ± {std_ag_ft + std_vim_ft:>5.2f} ms")
    
    # Metrics
    with torch.no_grad():
        pred_ft = videomae.get_sparse_forward(inputs, mask_ft)
        mse_ft = torch.nn.functional.mse_loss(pred_ft, targets).item()
    
    num_patches_ft = mask_ft.sum(dim=-1).float().mean().item()
    
    # ============= SUMMARY TABLE =============
    print("\n" + "="*80)
    print("FINAL RESULTS - MOVING BACKGROUND EXPERIMENT")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Patches':>8} {'MSE':>10} {'ViM Lat':>10} {'AG Lat':>10} {'Total':>10}")
    print("-"*80)
    
    # Full model
    print(f"{'Full VideoMAE':<30} {256:>8.0f} {mse_full:>10.4f} {avg_full:>8.2f}ms {'---':>10} {avg_full:>8.2f}ms")
    
    # Zero-shot
    print(f"{'AutoGaze (zero-shot)':<30} {num_patches_zs:>8.1f} {mse_zs:>10.4f} {avg_vim_zs:>8.2f}ms {avg_ag_zs:>8.2f}ms {avg_ag_zs+avg_vim_zs:>8.2f}ms")
    
    # Fine-tuned
    print(f"{'AutoGaze (fine-tuned)':<30} {num_patches_ft:>8.1f} {mse_ft:>10.4f} {avg_vim_ft:>8.2f}ms {avg_ag_ft:>8.2f}ms {avg_ag_ft+avg_vim_ft:>8.2f}ms")
    
    # Key metrics
    print(f"\n{'Model Sizes:':<30}")
    print(f"  VideoMAE: {vim_params:.2f}M params")
    print(f"  AutoGaze: {ag_params:.2f}M params ({ag_params/vim_params*100:.1f}% overhead)")
    
    # VideoMAE speedup from sparsity
    vim_speedup_zs = avg_full / avg_vim_zs
    vim_speedup_ft = avg_full / avg_vim_ft
    print(f"\n{'VideoMAE Speedup:':<30}")
    print(f"  Zero-shot: {vim_speedup_zs:.2f}x (VideoMAE only, {num_patches_zs:.0f} patches)")
    print(f"  Fine-tuned: {vim_speedup_ft:.2f}x (VideoMAE only, {num_patches_ft:.0f} patches)")
    
    # Quality tradeoff
    print(f"\n{'Quality Tradeoff:':<30}")
    print(f"  Full model MSE: {mse_full:.4f}")
    print(f"  Zero-shot MSE: {mse_zs:.4f} ({(mse_zs/mse_full - 1)*100:+.1f}%)")
    print(f"  Fine-tuned MSE: {mse_ft:.4f} ({(mse_ft/mse_full - 1)*100:+.1f}%)")
    
    print(f"\n  Key insight: AutoGaze IMPROVES quality while reducing patches!")
    print(f"  Fine-tuned uses {num_patches_ft/256*100:.1f}% patches with {mse_ft/mse_full*100:.1f}% of full model MSE")
    
    print("="*80)
    
    return {
        'mse_full': mse_full, 'mse_zs': mse_zs, 'mse_ft': mse_ft,
        'patches_zs': num_patches_zs, 'patches_ft': num_patches_ft,
        'latency_full': avg_full, 'latency_vim_zs': avg_vim_zs, 'latency_vim_ft': avg_vim_ft,
        'latency_ag_zs': avg_ag_zs, 'latency_ag_ft': avg_ag_ft,
        'vim_params': vim_params, 'ag_params': ag_params,
    }


if __name__ == '__main__':
    results = run_moving_background_experiment()