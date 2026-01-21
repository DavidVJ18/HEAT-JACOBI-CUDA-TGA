#include "heat.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Definición para el tamaño del bloque (número de threads por bloque 1D)
#define BLOCK_SIZE_X 256

// Definición para el tamaño del bloque 2D (usado en BLOCK_WORK)
#define BLOCK_SIZE_BLOCK_KERNEL_X 16
#define BLOCK_SIZE_BLOCK_KERNEL_Y 16

#define ROW_WORK 0
#define COLUMN_WORK 1
#define BLOCK_WORK 2
#define THREADWORK BLOCK_WORK
#define SHARED 0

// --- KERNELS ACTUALIZADOS CON OFFSET ---

/**
 * Kernel Jacobi por FILAS (Soporta Offset para Streams)
 */
__global__ void jacobi_row_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs, int i_offset) {
    // Sumamos el offset al índice global
    int i = (blockIdx.x * blockDim.x + threadIdx.x) + i_offset;

    if (i >= 1 && i < sizex - 1) {
        double local_diff_sum = 0.0;
        double tmp;
        for (int j = 1; j < sizey - 1; j++) {
            int center = i * sizey + j;
            int left   = i * sizey + (j - 1);
            int right  = i * sizey + (j + 1);
            int top    = (i - 1) * sizey + j;
            int bottom = (i + 1) * sizey + j;

            tmp = 0.25 * (u[left] + u[right] + u[top] + u[bottom]);
            double diff = tmp - u[center];
            local_diff_sum += diff * diff;
            unew[center] = tmp;
        }
        diffs[i] = local_diff_sum;
    } else {
        if (i < sizex) diffs[i] = 0.0; 
    }
}

/**
 * Kernel Jacobi por FILAS con SHARED MEMORY (Soporta Offset)
 */
__global__ void jacobi_shared_row_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs, int i_offset) {
    __shared__ double s_col[BLOCK_SIZE_X + 2];
    int tid = threadIdx.x;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) + i_offset; // Offset añadido
    int s_idx = tid + 1;
    double local_diff_sum = 0.0;
    bool is_active = (i >= 1 && i < sizex - 1);

    for (int j = 1; j < sizey - 1; j++) {
        if (i < sizex) s_col[s_idx] = u[i * sizey + j];
        if (tid == 0 && i > 0) s_col[0] = u[(i - 1) * sizey + j];
        if (tid == blockDim.x - 1 && i < sizex - 1) s_col[BLOCK_SIZE_X + 1] = u[(i + 1) * sizey + j];
        
        __syncthreads();
        
        if (is_active) {
            int center_idx = i * sizey + j;
            double top_val = s_col[s_idx - 1];
            double bottom_val = s_col[s_idx + 1];
            double left_val = u[i * sizey + (j - 1)];
            double right_val = u[i * sizey + (j + 1)];
            double center_val = s_col[s_idx];
            
            double tmp = 0.25 * (left_val + right_val + top_val + bottom_val);
            double diff = tmp - center_val;
            local_diff_sum += diff * diff;
            unew[center_idx] = tmp;
        }
        __syncthreads();
    }
    if (i < sizex) diffs[i] = (is_active) ? local_diff_sum : 0.0;
}

/**
 * Kernel Jacobi por COLUMNAS (Soporta Offset)
 */
__global__ void jacobi_column_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs, int j_offset) {
    int j = (blockIdx.x * blockDim.x + threadIdx.x) + j_offset; // Offset añadido

    if (j >= 1 && j < sizey - 1) {
        double local_diff_sum = 0.0;
        double tmp;
        for (int i = 1; i < sizex - 1; i++) {
            int center = i * sizey + j;
            int left   = i * sizey + (j - 1);
            int right  = i * sizey + (j + 1);
            int top    = (i - 1) * sizey + j;
            int bottom = (i + 1) * sizey + j;

            tmp = 0.25 * (u[left] + u[right] + u[top] + u[bottom]);
            double diff = tmp - u[center];
            local_diff_sum += diff * diff;
            unew[center] = tmp;
        }
        diffs[j] = local_diff_sum;
    } else {
        if (j < sizey) diffs[j] = 0.0; 
    }
}

/**
 * Kernel Jacobi por COLUMNAS con SHARED MEMORY (Soporta Offset)
 */
__global__ void jacobi_shared_column_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs, int j_offset) {
    __shared__ double s_row[BLOCK_SIZE_X + 2];
    int tid = threadIdx.x;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) + j_offset; // Offset añadido
    int s_idx = tid + 1;
    double local_diff_sum = 0.0;
    bool is_active = (j >= 1 && j < sizey - 1);

    for (int i = 1; i < sizex - 1; i++) {
        if (j < sizey) s_row[s_idx] = u[i * sizey + j];
        if (tid == 0 && j > 0) s_row[0] = u[i * sizey + (j - 1)];
        if (tid == blockDim.x - 1 && j < sizey - 1) s_row[BLOCK_SIZE_X + 1] = u[i * sizey + (j + 1)];

        __syncthreads();

        if (is_active) {
            int center_idx = i * sizey + j;
            int top_idx = (i - 1) * sizey + j;
            int bottom_idx = (i + 1) * sizey + j;
            double left_val = s_row[s_idx - 1];
            double right_val = s_row[s_idx + 1];
            double top_val = u[top_idx];
            double bottom_val = u[bottom_idx];
            double center_val = s_row[s_idx];
            double tmp = 0.25 * (left_val + right_val + top_val + bottom_val);
            double diff = tmp - center_val;
            local_diff_sum += diff * diff;
            unew[center_idx] = tmp;
        }
        __syncthreads();
    }
    if (j < sizey) diffs[j] = (is_active) ? local_diff_sum : 0.0;
}

/**
 * Kernel Jacobi por BLOQUES con SHARED MEMORY y REDUCCIÓN
 * Acepta 'by_offset' para desplazar el índice de BLOQUE en el eje Y
 */
__global__ void jacobi_shared_block_kernel(double *u, double *unew, unsigned sizex, unsigned sizey, double *diffs, int by_offset) {
    
    // Configuración memoria compartida
    __shared__ double tile[BLOCK_SIZE_BLOCK_KERNEL_Y + 2][BLOCK_SIZE_BLOCK_KERNEL_X + 2];
    __shared__ double sdata[BLOCK_SIZE_BLOCK_KERNEL_X * BLOCK_SIZE_BLOCK_KERNEL_Y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y + by_offset; // APLICAMOS EL OFFSET DE STREAMS AQUÍ

    // Coordenadas globales
    int j = bx * blockDim.x + tx; 
    int i = by * blockDim.y + ty; 

    // Índice lineal dentro del bloque
    int tid = ty * blockDim.x + tx;

    // --- CARGA A SHARED MEMORY ---
    int row_in = i;
    int col_in = j;

    if (row_in < sizex && col_in < sizey) {
        tile[ty + 1][tx + 1] = u[row_in * sizey + col_in];
    }
    
    // Halo
    if (ty == 0 && row_in > 0) 
        tile[0][tx + 1] = u[(row_in - 1) * sizey + col_in];
    if (ty == BLOCK_SIZE_BLOCK_KERNEL_Y - 1 && row_in < sizex - 1) 
        tile[ty + 2][tx + 1] = u[(row_in + 1) * sizey + col_in];
    if (tx == 0 && col_in > 0) 
        tile[ty + 1][0] = u[row_in * sizey + (col_in - 1)];
    if (tx == BLOCK_SIZE_BLOCK_KERNEL_X - 1 && col_in < sizey - 1) 
        tile[ty + 1][tx + 2] = u[row_in * sizey + (col_in + 1)];

    __syncthreads();

    // --- CÁLCULO ---
    double local_error = 0.0;
    if (i >= 1 && i < sizex - 1 && j >= 1 && j < sizey - 1) {
        double tmp = 0.25 * (tile[ty][tx + 1] + tile[ty + 2][tx + 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 2]);
        unew[i * sizey + j] = tmp;
        double diff = tmp - tile[ty + 1][tx + 1];
        local_error = diff * diff;
    }

    // --- REDUCCIÓN ---
    sdata[tid] = local_error;
    __syncthreads();

    int n = BLOCK_SIZE_BLOCK_KERNEL_X * BLOCK_SIZE_BLOCK_KERNEL_Y;
    for (int s = n / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Escritura en memoria global (diffsBlock)
    if (tid == 0) {
        // ID único del bloque en el grid TOTAL
        int blockId = by * gridDim.x + bx;
        diffs[blockId] = sdata[0];
    }
}

// --- KERNEL REDUCCIÓN FINAL (Igual que antes) ---
__global__ void reduction_kernel07(double *g_idata, double *g_odata, int N) {
    __shared__ double sdata[BLOCK_SIZE_X];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < N) {
        double val1 = g_idata[i];
        double val2 = (i + blockDim.x < N) ? g_idata[i + blockDim.x] : 0.0;
        sdata[tid] += val1 + val2;
        i += gridSize;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile double *smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// --- FUNCIÓN SOLVER ---
double solve_cuda(algoparam_t *param, unsigned sizex, unsigned sizey, cudaStream_t *streams, int nStreams) {
    
    double *input_for_reduction = NULL;
    int elements_to_reduce = 0;
    
    // A) KERNEL JACOBI
    if(streams == NULL)
    {
        // --- SIN STREAMS (Pasamos Offset 0) ---
        if(THREADWORK == ROW_WORK)
        {
            dim3 block(BLOCK_SIZE_X);
            dim3 grid((sizex + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
            if(SHARED) jacobi_shared_row_kernel<<<grid, block>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, 0);
            else jacobi_row_kernel<<<grid, block>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, 0);
            
            input_for_reduction = param->diffs_dev;
            elements_to_reduce = sizex;
        }
        else if(THREADWORK == COLUMN_WORK)
        {      
            dim3 block(BLOCK_SIZE_X);
            dim3 grid((sizey + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
            if(SHARED) jacobi_shared_column_kernel<<<grid, block>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, 0);
            else jacobi_column_kernel<<<grid, block>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, 0);
            
            input_for_reduction = param->diffs_dev;
            elements_to_reduce = sizey;
        }
        else if(THREADWORK == BLOCK_WORK)
        {
            dim3 block(BLOCK_SIZE_BLOCK_KERNEL_X, BLOCK_SIZE_BLOCK_KERNEL_Y);
            dim3 grid((sizey + block.x - 1) / block.x, (sizex + block.y - 1) / block.y);
            
            jacobi_shared_block_kernel<<<grid, block>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffsBlock_dev, 0);
            
            input_for_reduction = param->diffsBlock_dev;
            elements_to_reduce = grid.x * grid.y; 
        }
    }
    else
    {
        // --- CON STREAMS ---
        if (THREADWORK == BLOCK_WORK) 
        {
            int total_block_rows = (sizex + BLOCK_SIZE_BLOCK_KERNEL_Y - 1) / BLOCK_SIZE_BLOCK_KERNEL_Y;
            int base = total_block_rows / nStreams;
            int rest = total_block_rows % nStreams;
            int offset_by = 0; 

            dim3 block(BLOCK_SIZE_BLOCK_KERNEL_X, BLOCK_SIZE_BLOCK_KERNEL_Y);
            int grid_x = (sizey + block.x - 1) / block.x; 

            for (int str = 0; str < nStreams; str++)
            {
                int nrows = base + (str < rest ? 1 : 0);
                if (nrows > 0) {
                    dim3 grid(grid_x, nrows);
                    jacobi_shared_block_kernel<<<grid, block, 0, streams[str]>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffsBlock_dev, offset_by);
                    offset_by += nrows;
                }
            }
            input_for_reduction = param->diffsBlock_dev;
            elements_to_reduce = total_block_rows * grid_x;
        }
        else 
        {
            int total_elems = (THREADWORK == ROW_WORK) ? sizex : sizey;
            int base = total_elems / nStreams;
            int rest = total_elems % nStreams;
            int offset = 0;
            dim3 block(BLOCK_SIZE_X);

            for (int str = 0; str < nStreams; str++)
            {
                int count = base + (str < rest ? 1 : 0);
                dim3 grid((count + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

                if (THREADWORK == ROW_WORK) {
                    if (SHARED) jacobi_shared_row_kernel<<<grid, block, 0, streams[str]>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, offset);
                    else jacobi_row_kernel<<<grid, block, 0, streams[str]>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, offset);
                }
                else { // COLUMN_WORK
                    if (SHARED) jacobi_shared_column_kernel<<<grid, block, 0, streams[str]>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, offset);
                    else jacobi_column_kernel<<<grid, block, 0, streams[str]>>>(param->u_dev, param->uhelp_dev, sizex, sizey, param->diffs_dev, offset);
                }
                offset += count;
            }
            input_for_reduction = param->diffs_dev;
            elements_to_reduce = total_elems;
        }
        
        for (int str = 0; str < nStreams; str++) cudaStreamSynchronize(streams[str]);
    }


    // B) KERNEL REDUCCIÓN
    int N = elements_to_reduce;
    
    int nThreads = 256; 
    int nBlocks = 1024; 
    
    if (N < nBlocks * nThreads * 2) {
        nBlocks = (N + (nThreads * 2 - 1)) / (nThreads * 2);
    }
    if (nBlocks < 1) nBlocks = 1;

    // Pasada 1
    reduction_kernel07<<<nBlocks, nThreads>>>(input_for_reduction, param->partial_sums_dev, N);

    // Pasada 2
    if (nBlocks > 1) {
        reduction_kernel07<<<1, nThreads>>>(param->partial_sums_dev, param->partial_sums_dev, nBlocks);
    }

    // Copiamos el resultado final
    double residual;
    cudaMemcpy(&residual, param->partial_sums_dev, sizeof(double), cudaMemcpyDeviceToHost);

    return residual;
}

void swap_mats_dev(algoparam_t *param) {    
    double *tmp = param->u_dev;
    param->u_dev = param->uhelp_dev;
    param->uhelp_dev = tmp;
}