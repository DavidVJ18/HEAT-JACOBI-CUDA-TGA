/*
 * heat.h
 * Global definitions for the iterative solver
 */
#ifndef HEAT_H
#define HEAT_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_GPUS 8 // Máximo número de GPUs soportadas

typedef struct
{
    float posx;
    float posy;
    float range;
    float temp;
}
heatsrc_t;

typedef struct
{
    unsigned maxiter;       
    unsigned resolution;    
    double   residual;      
    int      algorithm;     

    unsigned visres;        

    // Punteros Host (Pinned Memory)
    double *u, *uhelp;      
    double *uvis;
    double *diffs;
    
    // --- MULTI-GPU VARIABLES ---
    int numDevs; // Número de GPUs a usar
    
    // Arrays de punteros para gestionar N GPUs
    double *u_devs[MAX_GPUS];
    double *uhelp_devs[MAX_GPUS]; 
    double *diffsBlock_devs[MAX_GPUS];
    double *partial_sums_devs[MAX_GPUS];

    // Dimensiones locales por GPU
    unsigned local_sizex[MAX_GPUS]; // Altura local (incluyendo halos)
    unsigned local_sizey;           // Anchura (igual para todas, np)
    
    unsigned   numsrcs;     
    heatsrc_t *heatsrcs;
}
algoparam_t;

// function declarations
int initialize( algoparam_t *param );
int finalize( algoparam_t *param );
void write_image( FILE * f, double *u, unsigned sizex, unsigned sizey );
int coarsen(double *uold, unsigned oldx, unsigned oldy , double *unew, unsigned newx, unsigned newy );
int read_input( FILE *infile, algoparam_t *param );
void print_params( algoparam_t *param );
double wtime();

// Funciones del solver.cu
double solve_cuda_multi(algoparam_t *param);
void swap_mats_dev_multi(algoparam_t *param);
void halo_exchange_multi(algoparam_t *param);

#endif