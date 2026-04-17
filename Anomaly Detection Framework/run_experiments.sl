#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=ano_det_experiments_Anomaly_Detection_Text
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time 03:59:00
# --time 00:10:00
# --cpus-per-task 16

# environments                                                                                                                                                                                
# ---------------------------------                                                                                                                                                           
module purge
module load aidl/pytorch/2.5.1-cuda12.4
# ---------------------------------          

# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# JOB_DIR="jobs/job_${SLURM_JOB_ID}_${TIMESTAMP}"
# mkdir -p ${JOB_DIR}

echo "Job ID: ${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1


# srun python3 run_indep_anom.py --dataset_name imdb --runall --type_emb bert --nu 0.0 --nb_runs 6 --date
# srun python3 run_indep_anom.py --dataset_name enron --runall --type_emb bert --nu 0.0 --nb_runs 6 --cvdd
srun python3 run_indep_anom.py --dataset_name imdb --inlier_topic positive --type_emb bert --nu 0.0 --nb_runs 6 --fm_trans
# srun python3 create_tables_latex.py --type_embedding bert --nu 0.0 --all_metrics
