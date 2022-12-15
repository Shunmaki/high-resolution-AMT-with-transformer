# ============ Train piano transcription system from scratch ============
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/notebooks/maki/high-resolution-AMT/maestro-v3.0.0"

# Modify to your workspace
WORKSPACE="/notebooks/maki/high-resolution-AMT-with-transformer/workspaces"

#NOTE_CHECKPOINT_PATH="/notebooks/maki/high-resolution-AMT/bestmodel/140000_iterations.pth"

# --- 1. Train note transcription system ---
python3 pytorch/main.py train --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --loss_type='regress_onset_offset_frame_velocity_bce' --augmentation='none' --max_note_shift=0 --batch_size=16 --learning_rate=5e-4 --reduce_iteration=150000 --resume_iteration=0 --early_stop=15000 --cuda

# Inference probabiliy for evaluation
#python3 pytorch/calculate_score_for_paper.py infer_prob --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --checkpoint_path=$NOTE_CHECKPOINT_PATH --augmentation='none' --dataset='maestro' --split='test' --cuda

# Calculate metrics
#python3 pytorch/calculate_score_for_paper.py calculate_metrics --workspace=$WORKSPACE --model_type='Regress_onset_offset_frame_velocity_CRNN' --augmentation='none' --dataset='maestro' --split='test'