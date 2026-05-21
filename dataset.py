# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional

class MovingMNISTDataset(Dataset):
    """
    MovingMNIST dataset.
    
    The npy file has shape (20, 10000, 64, 64) where:
    - 20: number of frames per sequence
    - 10000: number of sequences
    - 64x64: frame resolution
    
    Each sequence contains 1-2 digits moving and bouncing off walls.
    """
    
    def __init__(
        self,
        data_path: str,
        input_frames: int = 10,
        pred_frames: int = 5,
        train: bool = True,
        num_train_videos: int = 8000,
        transform: bool = True,
        seed: int = 42,
    ):
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.total_frames = input_frames + pred_frames
        
        # Load data
        data = np.load(data_path)  # (20, 10000, 64, 64)
        data = data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Split train/val
        np.random.seed(seed)
        indices = np.random.permutation(data.shape[1])
        
        if train:
            indices = indices[:num_train_videos]
        else:
            indices = indices[num_train_videos:num_train_videos + 1000]
        
        self.data = data[:, indices, :, :]  # (20, N, 64, 64)
        self.data = np.transpose(self.data, (1, 0, 2, 3))  # (N, 20, 64, 64)
        
        # For sequences shorter than total_frames, we can use sliding windows
        self.num_sequences = self.data.shape[0]
        self.transform = transform
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get the full 20-frame sequence
        sequence = self.data[idx]  # (20, 64, 64)
        
        # Random starting point if sequence is longer than needed
        if sequence.shape[0] > self.total_frames:
            start = np.random.randint(0, sequence.shape[0] - self.total_frames)
            sequence = sequence[start:start + self.total_frames]
        
        # Split into input and target
        input_frames = sequence[:self.input_frames]  # (input_frames, 64, 64)
        target_frames = sequence[self.input_frames:self.total_frames]  # (pred_frames, 64, 64)
        
        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        input_frames = input_frames[:, np.newaxis, :, :]
        target_frames = target_frames[:, np.newaxis, :, :]
        
        return {
            'input': torch.FloatTensor(input_frames),
            'target': torch.FloatTensor(target_frames),
        }


class MovingMNISTWithBackground(Dataset):
    """
    MovingMNIST with a moving textured background.
    Creates a synthetic moving background and overlays MNIST digits.
    """
    
    def __init__(
        self,
        data_path: str,
        input_frames: int = 10,
        pred_frames: int = 5,
        train: bool = True,
        num_train_videos: int = 8000,
        img_size: int = 64,
        background_speed: int = 2,
        seed: int = 42,
    ):
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.total_frames = input_frames + pred_frames
        self.img_size = img_size
        self.background_speed = background_speed
        
        # Load base MNIST data
        data = np.load(data_path)
        data = data.astype(np.float32) / 255.0
        
        np.random.seed(seed)
        indices = np.random.permutation(data.shape[1])
        if train:
            indices = indices[:num_train_videos]
        else:
            indices = indices[num_train_videos:num_train_videos + 1000]
        
        self.data = data[:, indices, :, :]
        self.data = np.transpose(self.data, (1, 0, 2, 3))
        self.num_sequences = self.data.shape[0]
        
        # Generate a random textured background (larger than frame for panning)
        self.bg_size = img_size + 2 * background_speed * self.total_frames
        self._generate_background_texture()
    
    def _generate_background_texture(self):
        """Generate a random Perlin-like texture for the background."""
        np.random.seed(12345)  # Fixed seed for consistency
        # Create a larger background for panning
        x = np.linspace(0, 4*np.pi, self.bg_size)
        y = np.linspace(0, 4*np.pi, self.bg_size)
        X, Y = np.meshgrid(x, y)
        
        # Sum of multiple frequencies for texture
        self.bg_texture = (
            0.3 * np.sin(X) * np.cos(Y) +
            0.2 * np.sin(2*X) * np.cos(2*Y) +
            0.15 * np.sin(3*X + Y) +
            0.1 * np.cos(X - 2*Y)
        )
        self.bg_texture = (self.bg_texture - self.bg_texture.min()) / (self.bg_texture.max() - self.bg_texture.min())
        self.bg_texture = self.bg_texture * 0.3  # Dim background
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        if sequence.shape[0] > self.total_frames:
            start = np.random.randint(0, sequence.shape[0] - self.total_frames)
            sequence = sequence[start:start + self.total_frames]
        
        # Random background panning direction
        angle = np.random.uniform(0, 2*np.pi)
        dx = self.background_speed * np.cos(angle)
        dy = self.background_speed * np.sin(angle)
        
        input_frames = []
        target_frames = []
        
        for t in range(self.total_frames):
            # Extract moving crop from background
            start_x = int(self.bg_size // 2 - self.img_size // 2 + dx * t)
            start_y = int(self.bg_size // 2 - self.img_size // 2 + dy * t)
            bg_crop = self.bg_texture[start_x:start_x + self.img_size, 
                                      start_y:start_y + self.img_size]
            
            # Overlay MNIST digit (threshold at 0.1 to create mask)
            digit = sequence[t]
            mask = digit > 0.1
            frame = bg_crop.copy()
            frame[mask] = digit[mask]
            
            if t < self.input_frames:
                input_frames.append(frame)
            else:
                target_frames.append(frame)
        
        input_frames = np.stack(input_frames)[:, np.newaxis, :, :]
        target_frames = np.stack(target_frames)[:, np.newaxis, :, :]
        
        return {
            'input': torch.FloatTensor(input_frames),
            'target': torch.FloatTensor(target_frames),
        }


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    data_config = config.data_config
    
    if data_config.use_moving_background:
        DatasetClass = MovingMNISTWithBackground
        extra_kwargs = {
            'img_size': data_config.img_size,
            'background_speed': data_config.background_speed,
        }
    else:
        DatasetClass = MovingMNISTDataset
        extra_kwargs = {}
    
    train_dataset = DatasetClass(
        data_path=data_config.dataset_path,
        input_frames=data_config.input_frames,
        pred_frames=data_config.pred_frames,
        train=True,
        num_train_videos=data_config.num_train_videos,
        seed=config.train_config.seed,
        **extra_kwargs,
    )
    
    val_dataset = DatasetClass(
        data_path=data_config.dataset_path,
        input_frames=data_config.input_frames,
        pred_frames=data_config.pred_frames,
        train=False,
        num_train_videos=data_config.num_train_videos,
        seed=config.train_config.seed,
        **extra_kwargs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader