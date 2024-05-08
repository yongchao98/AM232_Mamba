#!/bin/bash
#SBATCH --job-name=14b_2gpu_rand
#SBATCH --partition=seas_gpu,gpu_requeue,serial_requeue,gpu
#SBATCH -n 16 # Number of cores
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100"
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime in D-HH:MM
#SBATCH --mem=32G # Memory pool for all cores in MB
#SBATCH -o rand_14b_2gpu_10ep.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e rand_14b_2gpu_10ep.err   # File to which STDERR will be written, %j inserts jobid


python Mamba_finetune.py
