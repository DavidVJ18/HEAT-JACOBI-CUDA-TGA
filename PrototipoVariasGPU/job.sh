#!/bin/bash

#SBATCH --job-name=HEAT_MULTI
#SBATCH -D .
#SBATCH --output=submit-heat-multi.o%j
#SBATCH --error=submit-heat-multi.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:4

export PATH=/Soft/cuda/12.2.2/bin:$PATH

RES=2046
ITER=250000

#echo "PRUEBA 1 GPU:"
#export CUDA_VISIBLE_DEVICES=0
#./heat.exe test.dat -s $RES -n $ITER > final.o

#echo "PRUEBA 2 GPUs:"
#export CUDA_VISIBLE_DEVICES=0,1
#./heat.exe test.dat -s $RES -n $ITER > final.o

echo "PRUEBA 4 GPUs:"
export CUDA_VISIBLE_DEVICES=0,1,2,3
./heat.exe test.dat -s $RES -n $ITER > final.o
