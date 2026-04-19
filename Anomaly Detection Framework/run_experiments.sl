#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=ano_det_experiments_Anomaly_Detection_Text
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time 01:00:00
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

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Written Work" --type_emb bert --nu 0.0 --nb_runs 5 --ocsvm --ae --rsrae --tccm
# srun python3 run_indep_anom.py --dataset_name dbpedia14 --runall --type_emb bert --nu 0.0 --nb_runs 5 --ocsvm --ae --rsrae --tccm
# srun python3 run_indep_anom.py --dataset_name dbpedia14 --runall --type_emb bert --nu 0.0 --nb_runs 3 --ocsvm --ae --rsrae --tccm --date --cvdd
# srun python3 run_indep_anom.py --dataset_name sst2 --inlier_topic positive --type_emb bert --nu 0.0 --nb_runs 6 --date
# srun python3 run_indep_anom.py --dataset_name enron --runall --type_emb bert --nu 0.0 --nb_runs 6 --cvdd
# srun python3 run_indep_anom.py --dataset_name sst2 --inlier_topic positive --type_emb bert --nu 0.0 --nb_runs 3 --fm_trans
# srun python3 create_tables_latex.py --type_embedding bert --nu 0.0 --all_metrics


# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Company" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Educational Institution" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Artist" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd 

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Athlete" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Office Holder" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Mean Of Transportation" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Building" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Natural Place" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Village" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Animal" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Plant" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Album" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

# srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Film" --type_emb bert --nu 0.0 --nb_runs 4 --cvdd

srun python3 run_indep_anom.py --dataset_name dbpedia14 --inlier_topic "Written Work" --type_emb bert --nu 0.0 --nb_runs 4 --date
