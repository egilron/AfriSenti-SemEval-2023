#!/bin/bash
#SBATCH --job-name=model-eval
#SBATCH --mail-type=FAIL
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00     
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1

set -o errexit  # Recommended for easier debugging
source /etc/profile

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-datasets/2.3.2-foss-2021a-Python-3.9.5
# python -m venv ../venvs/tweetnlp --clear
source ../venvs/tweetnlp/bin/activate
export PATH="$PATH:/fp/homes01/u01/ec-egilron/.local/bin"
# pip install -r requirements_tnlp.txt
python XLM-T-based_final_submission.py
