#!/bin/bash                                                                                                                                                                                   

# Slurm submission script,                                                                                                                                                                    
# CRIHAN v 1.00 - Jan 2026                                                                                                                                                                    
# support@criann.fr                                                                                                                                                                           

# Job name                                                                                                                                                                                    
#SBATCH -J "run_experiments_Anomaly_Detection_Text"                                                                                                                                                                         

#SBATCH --output=jobs/%j/%x-%j.out                                                                                                                                                            
#SBATCH --error=jobs/%j/%x-%j.err                                                                                                                                                             

# GPUs architecture and number                                                                                                                                                                
# ----------------------------                                                                                                                                                                
# Partition (submission class)                                                                                                                                                                
#SBATCH --partition hpda                                                                                                                                                                       

# GPUs per compute node                                                                                                                                                                       
#   8 (maximum) for gpu                                                                                                                                                                       
#   8 (maximum) for hpda                                                                                                                                                                      
# SBATCH --gpus-per-node=1   
#SBATCH --nodes=1  
# SBATCH --gres=gpu:a100_1g.10gb                                                                                                                                                                 
# SBATCH --gpus=1                                                                                                                                                                              

# ----------------------------                                                                                                                                                                
# processes / tasks                                                                                                                                                                           
#SBATCH -n 1                                                                                                                                                                                  

# ----------------------------                                                                                                                                                                
# CPUs per task                                                                                                                                                                               
# Set the number of cpu in proportion to the number of GPU's devices :                                                                                                                        
#   gpu: until 8 cores / device                                                                                                                                                               
#   hpda: until 8 cores / device                                                                                                                                                              
#SBATCH --cpus-per-task 16                                                                                                                                                                     

# ------------------------                                                                                                                                                                    
# Job time (hh:mm:ss)                                                                                                                                                                         
#SBATCH --time 12:00:00                                                                                                                                                                       
# ------------------------                                                                                                                                                                    

##SBATCH --mail-type ALL                                                                                                                                                                      
# User e-mail address                                                                                                                                                                         
# # SBATCH --mail-user firstname.name@domain.ext  

# environments                                                                                                                                                                                
# ---------------------------------                                                                                                                                                           
module purge
module load aidl/pytorch/2.5.1-cuda12.4
# ---------------------------------          

# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# JOB_DIR="jobs/job_${SLURM_JOB_ID}_${TIMESTAMP}"
# mkdir -p ${JOB_DIR}

echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Timestamp: ${TIMESTAMP}"

export PYTHONUNBUFFERED=1

# srun bash run.sh 2>&1 | tee ${JOB_DIR}/run.log
srun python3 run_indep_anom.py --dataset_name 20newsgroups --runall --type_emb sentence-bert --nu 0.1 --cvdd
