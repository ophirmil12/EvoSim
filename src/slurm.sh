#!/bin/bash
#SBATCH --job-name=EvoSim
#SBATCH --killable
#SBATCH --requeue
#SBATCH --time=90:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH -o slurm.out.%A_%a.out


echo "Script Starting..."
source /cs/labs/dina/ophirmil12/miniforge3/etc/profile.d/conda.sh
conda activate project_env
cd /cs/labs/dina/ophirmil12/NirCourse/EvoSim/


export PYTHONPATH="src"

echo "Running CLI SERVINE..."
python -m servine.cli src/hiv_simulation.yaml

