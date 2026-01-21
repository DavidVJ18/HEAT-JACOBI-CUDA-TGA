/*
 * heat.h
 *
 * Global definitions for the iterative solver
 */
#ifndef HEAT_H
#define HEAT_H

#include <stdio.h>
#include <stdlib.h>

// configuration

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
    unsigned maxiter;       // maximum number of iterations
    unsigned resolution;    // spatial resolution
    double   residual;      // value for convergence
    int      algorithm;     // 0=>Jacobi, 1=>Gauss

    unsigned visres;        // visualization resolution

    // Punteros Host (CPU) - Asignados con malloc/calloc
    double *u, *uhelp;      
    double *uvis;
    double *diffs;
    
    // Punteros Device (GPU) - Asignados con cudaMalloc
    double *u_dev;
    double *uhelp_dev; 
    double *diffs_dev;    // Vector de residuos parciales en Device
    double *diffsBlock_dev;
    
    double *partial_sums_dev;
    
    unsigned   numsrcs;     // number of heat sources
    heatsrc_t *heatsrcs;
}
algoparam_t;

// function declarations

// misc.c
int initialize( algoparam_t *param );
int finalize( algoparam_t *param );
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey );
int coarsen(double *uold, unsigned oldx, unsigned oldy ,
	    double *unew, unsigned newx, unsigned newy );
int read_input( FILE *infile, algoparam_t *param );
void print_params( algoparam_t *param );
double wtime();

// Funciones del solver.cu (Interfaz Host para la GPU)
double solve_cuda(algoparam_t *param, unsigned sizex, unsigned sizey, cudaStream_t *streams, int nStreams);
void swap_mats_dev (algoparam_t *param);

#endif
