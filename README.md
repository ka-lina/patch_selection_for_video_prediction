# patch_selection_for_video_prediction
Neural method for patch selection in ViT encoder of a video prediction model.

```
# Stage 1 only: Train VideoMAE
python main.py --data_path mnist_test_seq.npy \
               --input_frames 10 \
               --pred_frames 5 \
               --batch_size 32 \
               --videomae_epochs 50 \
               --skip_autogaze

python eval_only.py --data_path mnist_test_seq.npy --save_dir ./checkpoints

# Stage 2 only (requires Stage 1 checkpoint):
python main.py --data_path mnist_test_seq.npy \
               --input_frames 10 \
               --pred_frames 5 \
               --batch_size 32 \
               --autogaze_epochs 30 \
               --skip_videomae

# Full pipeline:
python main.py --data_path mnist_test_seq.npy \
               --input_frames 10 \
               --pred_frames 5 \
               --batch_size 32 \
               --videomae_epochs 50 \
               --autogaze_epochs 30

# With moving background (Stage 2 feature):
python main.py --data_path mnist_test_seq.npy \
               --input_frames 10 \
               --pred_frames 5 \
               --batch_size 32 \
               --moving_bg \
               --save_dir ./checkpoints_moving_bg
```

```
# Normal training from scratch:
python main.py --data_path mnist_test_seq.npy

# Resume VideoMAE from checkpoint (continues from where it left off):
python main.py --data_path mnist_test_seq.npy --resume_videomae

# Resume AutoGaze from checkpoint:
python main.py --data_path mnist_test_seq.npy --skip_videomae --resume_autogaze

# Resume both:
python main.py --data_path mnist_test_seq.npy --resume_videomae --resume_autogaze

# Train for more epochs (change the epoch count):
python main.py --data_path mnist_test_seq.npy --resume_videomae --videomae_epochs 100

# Skip everything and just evaluate:
python main.py --data_path mnist_test_seq.npy --skip_videomae --skip_autogaze

# Force retrain from scratch (ignores checkpoints):
python main.py --data_path mnist_test_seq.npy --force_retrain
```


```
# 1. Fine-tune VideoMAE (30 min)
python finetune_videomae.py --epochs 5

# 2. Train AutoGaze with fine-tuned model (1-2 hours)
python main.py --data_path mnist_test_seq.npy \
    --skip_videomae \
    --autogaze_epochs 30 \
    --batch_size 8 \
    --save_dir ./checkpoints_v2

# 3. Evaluate
python eval_only.py --data_path mnist_test_seq.npy --save_dir ./checkpoints_v2
```

```
# 1. Download KTH dataset (google "KTH actions dataset download")
# 2. Prepare one class:
python prepare_kth.py --kth_dir ./KTH --class_name walking --output kth_walking.npy
# 3. Train:
python main.py --data_path kth_walking.npy \
    --videomae_epochs 100 \
    --batch_size 8 \
    --save_dir ./checkpoints_kth
```