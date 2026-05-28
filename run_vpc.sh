#!/bin/bash
#SBATCH --mem=50G
#SBATCH -J vpc
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -w vrhpc2.dsic.upv.es
#SBATCH --time=3-00:00
#SBATCH --cpus-per-task=4
#SBATCH -o vpc.log

set -e

# Edita solo esto si en el cluster cambia algun nombre o ruta.
CONDA_ENV="vpc_lab"
INSTALL_DEPS=1
RUN_VIT=0
RUN_ISIC=1
RUN_GENDER=0
RUN_CIFAR=0
ISIC_DATA_DIR="isic_segmentations"
GENDER_DATA_DIR="notebook"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install -r requirements.txt
fi

python - <<'PY'
import torch
import tensorflow as tf
print("CUDA torch:", torch.cuda.is_available())
print("GPUs tensorflow:", tf.config.list_physical_devices("GPU"))
PY

if [ "$RUN_VIT" = "1" ]; then
  python src/vit_attention_maps.py \
    --image-dir images \
    --out-dir results_vit_attention \
    --model-name vit_tiny_patch16_224 \
    --layers 0,3,6,11 \
    --heads mean \
    --device cuda \
    --rollout
fi

if [ "$RUN_ISIC" = "1" ]; then
  python src/isic_unet_segmentation.py \
    --data-dir "$ISIC_DATA_DIR" \
    --out-dir results_isic_unet \
    --epochs 30 \
    --batch-size 8 \
    --image-size 256 \
    --device cuda \
    --num-workers "$SLURM_CPUS_PER_TASK"
fi

if [ "$RUN_GENDER" = "1" ]; then
  python src/gender.py \
    --data-dir "$GENDER_DATA_DIR" \
    --model both \
    --epochs 50 \
    --batch-size 64 \
    --out-dir results_gender \
    --augment
fi

if [ "$RUN_CIFAR" = "1" ]; then
  python src/cifar_wideresnet.py \
    --depth 16 \
    --width 4 \
    --epochs 30 \
    --batch-size 128 \
    --augment \
    --out-dir results_cifar_wrn
fi
