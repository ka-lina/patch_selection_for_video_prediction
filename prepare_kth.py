# prepare_kth.py - Convert KTH to MovingMNIST format

import cv2
import numpy as np
import os
from tqdm import tqdm

def prepare_kth_class(kth_dir, class_name, output_path, img_size=64, 
                       input_frames=10, pred_frames=5):
    """
    Convert one KTH action class to MovingMNIST-like npy file.
    
    KTH structure: kth_dir/class_name/person_scenario_video.avi
    Output: numpy array (num_clips, total_frames, img_size, img_size)
    """
    total_frames = input_frames + pred_frames
    all_clips = []
    
    class_path = os.path.join(kth_dir, class_name)
    if not os.path.exists(class_path):
        print(f"Directory {class_path} not found!")
        return
    
    video_files = [f for f in os.listdir(class_path) if f.endswith('.avi')]
    print(f"Found {len(video_files)} videos for {class_name}")
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            frames.append(resized)
        
        cap.release()
        frames = np.array(frames, dtype=np.float32) / 255.0
        
        # Extract sliding windows
        stride = max(1, (len(frames) - total_frames) // 20)  # ~20 clips per video
        for start in range(0, len(frames) - total_frames + 1, stride):
            clip = frames[start:start + total_frames]
            if len(clip) == total_frames:
                all_clips.append(clip)
    
    # Convert to numpy array and save
    all_clips = np.array(all_clips)  # (N, total_frames, H, W)
    all_clips = np.transpose(all_clips, (1, 0, 2, 3))  # (T, N, H, W) - MovingMNIST format
    
    np.save(output_path, all_clips)
    print(f"Saved {all_clips.shape[1]} clips to {output_path}")
    print(f"Shape: {all_clips.shape}")

# Usage:
# prepare_kth_class('/path/to/KTH', 'walking', 'kth_walking.npy')