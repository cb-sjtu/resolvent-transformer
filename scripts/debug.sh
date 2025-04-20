#!/bin/bash
#PBS -P CFP01-SF-009
#PBS -j oe
#PBS -k oed
#PBS -N debug
#PBS -l walltime=30:00:00
#PBS -l select=1:ngpus=2
##----- CPU/Mem will be allocated at 10/200gb per GPU. -----
##----- sample config for ngpus of 2, 4, 8, 16 via either line below ----
###PBS -l select=1:ngpus=2
###PBS -l select=1:ngpus=4
###PBS -l select=1:ngpus=8
###PBS -l select=2:ngpus=8

cd $PBS_O_WORKDIR;

source ~/.bashrc
conda activate sg

python3 src/train.py --config-name=train_nop trainer.max_steps=100 trainer.val_check_interval=50 trainer.limit_val_batches=50
python3 src/train.py --config-name=train_vicon trainer.max_steps=100 trainer.val_check_interval=50 trainer.limit_val_batches=50


echo "Done"

##**************************************************************************
##   WARNING and IMPORTANT NOTICE                                          *
##**************************************************************************
##   DON'T SET  [CUDA_VISIBLE_DEVICES]  in your Python Program!            *
##   PBS Job Scheduler will set GPU devices for the job automatically.     *
##**************************************************************************
