#!/bin/bash
#SBATCH --mem=50G              # Total Memory
#SBATCH -J exist               # Job name
#SBATCH -N 1                   # Amount of nodes
#SBATCH -G 1                   # Num of GPUs
#SBATCH -w vrhpc2.dsic.upv.es  # Node to run
#SBATCH --time=3-00:00         # Time, Days-HH:MM format
#SBATCH --cpus-per-task=4      # Total cores
#SBATCH -o alc_l2.log            # STDOUT


# Activar Conda
eval "$(conda shell.bash hook)"
conda activate alc_lab2

# Ejecutar el script de Python
mkdir -p runs/search_cpu

for seed in 13 21 42 77 101; do
  for clip in "0.5 99.5" "1.0 99.0" "2.0 98.0"; do
    for pca in 0.90 0.95 0.99; do
      read low high <<< "$clip"
      out="runs/search_cpu/s${seed}_c${low}-${high}_p${pca}"
      python run_lab2_exist2026_task2.py \
        --train_json lab2_materials/dataset_task2_exist2026/training.json \
        --train_golds lab2_materials/golds_task2_exist2026/training_golds.json \
        --train_blip_csv lab2_materials/dataset_task2_exist2026/blip_captions_training.csv \
        --test_json lab2_materials/dataset_task2_exist2026/test.json \
        --test_blip_csv lab2_materials/dataset_task2_exist2026/blip_captions_test.csv \
        --use_mami \
        --mami_train_csv mami_dataset/training_mami.csv \
        --seed "$seed" \
        --top_k 3 \
        --eeg_clip_low_pct "$low" \
        --eeg_clip_high_pct "$high" \
        --eeg_var_threshold 1e-6 \
        --eeg_pca_variance "$pca" \
        --output_dir "$out"
        --use_clip \
        --memes_root _OPTIONAL__Memes_of_EXIST2024/memes_training_part1 \
        --clip_device cuda \
        --output_dir runs/search_clip/mi_run
    done
  done
done
