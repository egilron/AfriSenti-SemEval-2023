#!/bin/bash
#SBATCH --job-name=afro-xlmr
#SBATCH --mail-type=FAIL
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00     
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
echo $0
# functions = {train_lg_groups,  train_individual}
echo "xlmt_finetuning3.py"
python xlmt_finetuning3.py $1
