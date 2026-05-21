# config.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class DataConfig:
    """Configuration for MovingMNIST dataset."""
    dataset_path: str = "mnist_test_seq.npy"
    img_size: int = 64
    input_frames: int = 10       # Number of input frames
    pred_frames: int = 5         # Number of frames to predict
    batch_size: int = 32
    num_workers: int = 4
    num_train_videos: int = 8000  # Number of training videos
    num_val_videos: int = 1000    # Number of validation videos
    use_moving_background: bool = False  # Stage 2 feature
    background_speed: int = 2    # Pixels per frame

# @dataclass
# class VideoMAEConfig:
#     """Configuration for the Predictive VideoMAE."""
#     img_size: int = 64
#     patch_size: int = 8          # 8x8 patches
#     in_channels: int = 1         # Grayscale MNIST
#     embed_dim: int = 256
#     depth: int = 6               # Number of transformer blocks in encoder
#     num_heads: int = 8
#     decoder_depth: int = 3       # Decoder is smaller
#     decoder_num_heads: int = 4
#     mlp_ratio: float = 4.0
#     mask_ratio: float = 0.75     # For pretraining (not used in prediction mode)
    
#     @property
#     def num_patches(self) -> int:
#         return (self.img_size // self.patch_size) ** 2  # 64 for 8x8 patches on 64x64
    
#     @property
#     def patch_dim(self) -> int:
#         return self.in_channels * self.patch_size * self.patch_size

@dataclass
class VideoMAEConfig:
    """Configuration for the Predictive VideoMAE."""
    img_size: int = 64
    patch_size: int = 4
    in_channels: int = 1
    embed_dim: int = 128          # Ultra-lean
    depth: int = 4                # Minimal transformer
    num_heads: int = 4            # 128/4 = 32 dim per head
    decoder_depth: int = 3        # Kept for compatibility
    decoder_num_heads: int = 4    # Kept for compatibility
    mlp_ratio: float = 2.0        # Smaller FFN
    mask_ratio: float = 0.75
    pred_frames: int = 5
    
    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size


# @dataclass
# class AutoGazeConfig:
#     """Simplified AutoGaze configuration."""
#     img_size: int = 64
#     patch_size: int = 8
#     hidden_dim: int = 128
#     num_scales: int = 3          # Multi-scale patches: 8, 16, 32
#     gaze_decoder_layers: int = 2
#     gaze_decoder_heads: int = 4
#     max_patches_per_frame: int = 16  # Maximum patches to select per frame
#     temperature: float = 0.1
#     use_multi_scale: bool = True
    
#     @property
#     def num_patches(self) -> int:
#         return (self.img_size // self.patch_size) ** 2

@dataclass
class AutoGazeConfig:
    """Simplified AutoGaze configuration."""
    img_size: int = 64
    patch_size: int = 4          # Match VideoMAE's patch size!
    hidden_dim: int = 128
    num_scales: int = 3
    gaze_decoder_layers: int = 2
    gaze_decoder_heads: int = 4
    max_patches_per_frame: int = 64  # Increased from 16 since we have 256 patches now
    temperature: float = 0.1
    use_multi_scale: bool = True
    
    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2  # 256 for 4x4 patches

@dataclass
class TrainConfig:
    """Training configuration."""
    # Stage 1: VideoMAE
    videomae_lr: float = 1e-3
    videomae_epochs: int = 50
    videomae_warmup_epochs: int = 5
    
    # Stage 2: AutoGaze
    autogaze_lr: float = 1e-4
    autogaze_epochs: int = 30
    autogaze_warmup_epochs: int = 3
    autogaze_rl_epochs: int = 10  # Optional RL fine-tuning
    
    # General
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    save_dir: str = "./checkpoints"
    log_interval: int = 50
    device: str = "cuda"
    seed: int = 42