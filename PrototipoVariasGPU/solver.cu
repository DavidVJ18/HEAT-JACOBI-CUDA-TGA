#include "heat.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// --- KERNEL (Bloques + Shared + Reducción) ---
__global__ void jacobi_shared_block_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs) {
    
    __shared__ double tile[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
    __shared__ double sdata[BLOCK_SIZE_X * BLOCK_SIZE_Y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int j = bx * blockDim.x + tx; 
    int i = by * blockDim.y + ty; 
    int tid = ty * blockDim.x + tx;

    // Carga a Shared Memory con Halo
    int row_in = i;
    int col_in = j;

    if (row_in < sizex && col_in < sizey) {
        tile[ty + 1][tx + 1] = u[row_in * sizey + col_in];
    }
    
    // Cargar Halos del bloque a Shared Memory
    if (ty == 0 && row_in > 0) tile[0][tx + 1] = u[(row_in - 1) * sizey + col_in];
    if (ty == BLOCK_SIZE_Y - 1 && row_in < sizex - 1) tile[ty + 2][tx + 1] = u[(row_in + 1) * sizey + col_in];
    if (tx == 0 && col_in > 0) tile[ty + 1][0] = u[row_in * sizey + (col_in - 1)];
    if (tx == BLOCK_SIZE_X - 1 && col_in < sizey - 1) tile[ty + 1][tx + 2] = u[row_in * sizey + (col_in + 1)];

    __syncthreads();

    // Cálculo Jacobi
    double local_error = 0.0;
    // IMPORTANTE: Evitamos procesar los bordes GLOBALES o HALOS de la GPU (fila 0 y fila sizex-1)
    if (i >= 1 && i < sizex - 1 && j >= 1 && j < sizey - 1) {
        double tmp = 0.25 * (tile[ty][tx + 1] + tile[ty + 2][tx + 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 2]);
        unew[i * sizey + j] = tmp;
        double diff = tmp - tile[ty + 1][tx + 1];
        local_error = diff * diff;
    }

    // Reducción parcial en el bloque
    sdata[tid] = local_error;
    __syncthreads();

    int n = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        int blockId = by * gridDim.x + bx;
        diffs[blockId] = sdata[0];
    }
}

// Kernel de reducción final
__global__ void reduction_kernel(double *g_idata, double *g_odata, int N) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < N) ? g_idata[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// --- FUNCIÓN HALO EXCHANGE (Intercambio de fronteras entre GPUs) ---
void halo_exchange_multi(algoparam_t *param) {
    int np = param->local_sizey; // Ancho de la matriz

    for (int i = 0; i < param->numDevs; i++) {
        // Enviar mi fila superior real (fila 1) al vecino de ARRIBA (i-1)
        if (i > 0) {
            int my_row_idx = 1; // Mi primera fila real
            int neighbor_row_idx = param->local_sizex[i-1] - 1; // Su halo inferior
            
            // Copia Directa P2P
            cudaMemcpyPeer(
                param->u_devs[i-1] + neighbor_row_idx * np, i-1, // Destino
                param->u_devs[i] + my_row_idx * np, i,           // Origen
                np * sizeof(double)
            );
        }

        // Enviar mi fila inferior real (sizex - 2) al vecino de ABAJO (i+1)
        if (i < param->numDevs - 1) {
            int my_row_idx = param->local_sizex[i] - 2; // Mi última fila real
            int neighbor_row_idx = 0;                   // Su halo superior
            
            cudaMemcpyPeer(
                param->u_devs[i+1] + neighbor_row_idx * np, i+1, // Destino
                param->u_devs[i] + my_row_idx * np, i,           // Origen
                np * sizeof(double)
            );
        }
    }
}

// --- SOLVER MULTI-GPU ---
double solve_cuda_multi(algoparam_t *param) {
    double total_residual = 0.0;
    
    // 1. Lanzar Kernels en todas las GPUs
    for (int i = 0; i < param->numDevs; i++) {
        cudaSetDevice(i);
        
        unsigned sizex = param->local_sizex[i];
        unsigned sizey = param->local_sizey;
        
        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 grid((sizey + block.x - 1) / block.x, (sizex + block.y - 1) / block.y);

        // Llamada al kernel de cálculo
        jacobi_shared_block_kernel<<<grid, block>>>(
            param->u_devs[i], 
            param->uhelp_devs[i], 
            sizex, sizey, 
            param->diffsBlock_devs[i]
        );

        // Reducción del residuo (paso 1)
        int numBlocksKernel = grid.x * grid.y;
        int threadsRed = 256;
        int blocksRed = (numBlocksKernel + threadsRed - 1) / threadsRed;
        
        reduction_kernel<<<blocksRed, threadsRed, threadsRed * sizeof(double)>>>(
            param->diffsBlock_devs[i], 
            param->partial_sums_devs[i], 
            numBlocksKernel
        );
        
        // Reducción del residuo (paso 2 - final a 1 valor)
        reduction_kernel<<<1, threadsRed, threadsRed * sizeof(double)>>>(
            param->partial_sums_devs[i], 
            param->partial_sums_devs[i], 
            blocksRed
        );
    }

    // 2. Recoger residuos de todas las GPUs
    for (int i = 0; i < param->numDevs; i++) {
        cudaSetDevice(i);
        double local_res = 0.0;
        cudaMemcpy(&local_res, param->partial_sums_devs[i], sizeof(double), cudaMemcpyDeviceToHost);
        total_residual += local_res;
    }

    return total_residual;
}

// Intercambia punteros en TODAS las GPUs
void swap_mats_dev_multi(algoparam_t *param) {
    for(int i=0; i<param->numDevs; i++) {
        double *tmp = param->u_devs[i];
        param->u_devs[i] = param->uhelp_devs[i];
        param->uhelp_devs[i] = tmp;
    }
}