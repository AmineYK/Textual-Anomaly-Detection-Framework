#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=ano_det_experiments_Anomaly_Detection_Text
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
#SBATCH --partition=gpu
# SBATCH --partition=hpda_mig
# SBATCH --partition=hpda
# SBATCH --nodes=1
#SBATCH --gres=gpu:1
# SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time 02:50:00
# SBATCH --array=0-13

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
# topics=('trade' 'money-fx' 'crude' 'ship' 'interest')
# topics=('computer' 'recreation' 'science' 'miscellaneous' 'politics' 'religion')
# topics=('negative' 'positive')
# topics=("Building" "Natural Place" "Village" "Animal" "Plant" "Album" "Film" "Written Work" \
#     "Company" "Educational Institution" "Artist" "Athlete" "Office Holder" "Mean Of Transportation")
# topics=("World" "Business" "Sports" "Sci-Tech")
# topics=("World" "Business" "Sci-Tech")

# topic=${topics[$SLURM_ARRAY_TASK_ID]}

# srun python3 run_sentence_level.py \
#     --dataset_name dbpedia14 \
#     --inlier_topic "$topic" \
#     --type_emb e5 \
#     --nu 0.0 \
#     --nb_runs 4 \
#     --fm_trans
    # --ae --ocsvm --rsrae --tccm




# srun python3 run_sentence_level.py --dataset_name imdb --runall --type_emb st5 --nu 0.0 --nb_runs 5 --rsrae --ae --ocsvm --tccm
# srun python3 run_sentence_level.py --dataset_name sst2 --inlier_topic "positive" --type_emb st5 --nu 0.0 --nb_runs 5 --fm_trans

# srun python3 run_sentence_level.py --dataset_name dbpedia14 --runall --type_emb st5 --nu 0.0 --nb_runs 5 --rsrae --ae --ocsvm --tccm
# srun python3 run_sentence_level.py --dataset_name imdb --inlier_topic "positive" --type_emb st5 --nu 0.0 --nb_runs 5 --fm_trans
# srun python3 run_sentence_level.py --dataset_name reuters --runall --type_emb mpnet --nu 0.0 --nb_runs 6 --fm_trans
# srun python3 run_embedding.py --split "train"


# srun python3 run_sentence_level.py --dataset_name reuters --runall --type_emb distilroberta --nu 0.0 --nb_runs 6 --fm_trans
# srun python3 run_sentence_level.py --dataset_name dbpedia14 --inlier_topic "Company" --type_emb mpnet --nu 0.0 --nb_runs 4 --fate



# srun python3 run_token_level.py --dataset_name reuters --inlier_topic "trade" --type_emb modernbert --nu 0.0 --nb_runs 4 --seq_len 128 --cvdd --date
# srun python3 run_token_level.py --dataset_name imdb --inlier_topic "negative" --type_emb qwen --nu 0.0 --nb_runs 2 --seq_len 256 --batch_size 256 --fm_trans

# srun python3 run_sentence_level.py --dataset_name sst2 --runall --type_emb e5 --nu 0.0 --nb_runs 4 --rsrae --ae --ocsvm --tccm
srun python3 run_sentence_level.py --dataset_name sst2 --inlier_topic "negative" --type_emb e5 --nu 0.0 --nb_runs 5 --fm_trans

# srun python3 run_experiments.py 