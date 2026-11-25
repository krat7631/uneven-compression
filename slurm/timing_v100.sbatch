#!/bin/bash
# Timing study on one V100 (courses-gpu), inside the course container.
#   sbatch slurm/timing_v100.sbatch            # all 16 settings
#   sbatch slurm/timing_v100.sbatch --quick    # dry run: 2 trials on 256 texts
# Extra arguments are passed to scripts/run_timing.py.
# Submit from the repo root (~/uneven-compression); results land in results/timing/<job id>/.
#SBATCH --job-name=uc-timing
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
apptainer exec --nv "$SIF" bash -c "source activate eai && cd $SLURM_SUBMIT_DIR && PYTHONPATH=src python scripts/run_timing.py $*"
