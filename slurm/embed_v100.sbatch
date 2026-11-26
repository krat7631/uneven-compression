#!/bin/bash
# Embed the evaluation data at all four precisions on one V100 (quality study; scoring runs on the Mac).
#   sbatch slurm/embed_v100.sbatch             # all four precisions
#   sbatch slurm/embed_v100.sbatch --limit 300 # dry run
# Extra arguments are passed to scripts/embed_precisions.py.
# Submit from the repo root (~/uneven-compression); results land in results/embeddings/<job id>/.
#SBATCH --job-name=uc-embed
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

source /etc/profile.d/modules.sh
module use /shared/EL9/explorer/modulefiles
module load explorer cuda/12.1.1

SIF=/shared/container_repository/explorer/courses/jupyterlab/jupyterlab_tf_pt_nividia/jupyterlab_tf_pt_nvidia_cudnn_xgboost.sif
export USE_TF=0
export TORCHINDUCTOR_COMPILE_THREADS=1

echo "node: $(hostname)  job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# `source activate eai` gives the container's torch; PYTHONPATH=src uses this repo's code without installing it.
apptainer exec --nv "$SIF" bash -c "source activate eai && cd $SLURM_SUBMIT_DIR && PYTHONPATH=src python scripts/embed_precisions.py $*"
