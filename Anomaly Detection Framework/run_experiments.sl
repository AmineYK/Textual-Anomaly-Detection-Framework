#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=ano_det_experiments_Anomaly_Detection_Text
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
# SBATCH --partition=gpu
#SBATCH --partition=hpda_mig
# SBATCH --partition=hpda
# SBATCH --nodes=1
# SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time 01:20:00
#SBATCH --array=0-13

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

# topics=('earn' 'trade' 'acq' 'money-fx' 'crude' 'ship' 'interest')
# topics=('computer' 'recreation' 'science' 'miscellaneous' 'politics' 'religion')
topics=("Building" "Natural Place" "Village" "Animal" "Plant" "Album" "Film" "Written Work" \
    "Company" "Educational Institution" "Artist" "Athlete" "Office Holder" "Mean Of Transportation")
# # topics=("World" "Business" "Sports" "Sci-Tech")

topic=${topics[$SLURM_ARRAY_TASK_ID]}

srun python3 run_sentence_level.py \
    --dataset_name dbpedia14 \
    --inlier_topic "$topic" \
    --type_emb distilroberta \
    --nu 0.0 \
    --nb_runs 3 \
    --fate




# srun python3 run_sentence_level.py --dataset_name imdb --runall --type_emb st5 --nu 0.0 --nb_runs 5 --rsrae --ae --ocsvm --tccm
# srun python3 run_sentence_level.py --dataset_name sst2 --inlier_topic "positive" --type_emb st5 --nu 0.0 --nb_runs 5 --fm_trans

# srun python3 run_sentence_level.py --dataset_name dbpedia14 --runall --type_emb st5 --nu 0.0 --nb_runs 5 --rsrae --ae --ocsvm --tccm
# srun python3 run_sentence_level.py --dataset_name imdb --inlier_topic "positive" --type_emb st5 --nu 0.0 --nb_runs 5 --fm_trans
# srun python3 run_sentence_level.py --dataset_name reuters --runall --type_emb mpnet --nu 0.0 --nb_runs 6 --fm_trans
# srun python3 run_embedding.py --split "train"


# srun python3 run_sentence_level.py --dataset_name reuters --runall --type_emb distilroberta --nu 0.0 --nb_runs 6 --fm_trans
# srun python3 run_sentence_level.py --dataset_name dbpedia14 --inlier_topic "Company" --type_emb distilroberta --nu 0.0 --nb_runs 3 --fate

