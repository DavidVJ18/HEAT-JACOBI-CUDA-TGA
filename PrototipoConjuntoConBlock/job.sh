#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=HEAT_CUDA
#SBATCH -D .
#SBATCH --output=submit-heat.o%j
#SBATCH --error=submit-heat.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## Selecciona la GPU (He dejado activa la opción de 1 RTX 3080 como en tu ejemplo)

## OPCION A: Usamos la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080 (Opción por defecto)
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1

export PATH=/Soft/cuda/12.2.2/bin:$PATH


echo "2" > test.dat
echo "0.0 0.0 1.0 2.5" >> test.dat
echo "0.5 1.0 1.0 2.5" >> test.dat


echo "Iniciando ejecución normal..."
./heat.exe test.dat 

echo "Ejecucion finalizada."
