/*
 * misc.cu
 * Helper functions
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "heat.h"

// ACTIVAMOS PINNED MEMORY
#define PINNED 1

int initialize( algoparam_t *param )
{
    int i, j;
    double dist;

    // total number of points (including border)
    const int np = param->resolution + 2;

    // allocate memory
    if(PINNED){
        cudaMallocHost((double**)&(param->u),sizeof(double)*np*np);
        cudaMallocHost((double**)&(param->uhelp),sizeof(double)*np*np);
        cudaMallocHost((double**)&(param->uvis),sizeof(double)*(param->visres+2)*(param->visres+2));
        cudaMallocHost((double**)&(param->diffs),sizeof(double)*np);
    }
    else {
        (param->u)     = (double*)calloc( sizeof(double),np*np );
        (param->uhelp) = (double*)calloc( sizeof(double),np*np );
        (param->uvis)  = (double*)calloc( sizeof(double), (param->visres+2)*(param->visres+2) );
        (param->diffs) = (double*)calloc( sizeof(double), np );
    }

    if( !(param->u) || !(param->uhelp) || !(param->uvis) || !(param->diffs)) {
        fprintf(stderr, "Error: Cannot allocate memory\n");
        return 0;
    }

    for( i=0; i<param->numsrcs; i++ ) {
        /* top row */
        for( j=0; j<np; j++ ) {
            dist = sqrt( pow((double)j/(double)(np-1) - param->heatsrcs[i].posx, 2)+
                         pow(param->heatsrcs[i].posy, 2));
            if( dist <= param->heatsrcs[i].range ) {
                (param->u)[j] += (param->heatsrcs[i].range-dist) / param->heatsrcs[i].range * param->heatsrcs[i].temp;
            }
        }
        /* bottom row */
        for( j=0; j<np; j++ ) {
            dist = sqrt( pow((double)j/(double)(np-1) - param->heatsrcs[i].posx, 2)+
                         pow(1-param->heatsrcs[i].posy, 2));
            if( dist <= param->heatsrcs[i].range ) {
                (param->u)[(np-1)*np+j]+= (param->heatsrcs[i].range-dist) / param->heatsrcs[i].range * param->heatsrcs[i].temp;
            }
        }
        /* leftmost column */
        for( j=1; j<np-1; j++ ) {
            dist = sqrt( pow(param->heatsrcs[i].posx, 2)+
                         pow((double)j/(double)(np-1) - param->heatsrcs[i].posy, 2));
            if( dist <= param->heatsrcs[i].range ) {
                (param->u)[ j*np ]+= (param->heatsrcs[i].range-dist) / param->heatsrcs[i].range * param->heatsrcs[i].temp;
            }
        }
        /* rightmost column */
        for( j=1; j<np-1; j++ ) {
            dist = sqrt( pow(1-param->heatsrcs[i].posx, 2)+
                         pow((double)j/(double)(np-1) - param->heatsrcs[i].posy, 2));
            if( dist <= param->heatsrcs[i].range ) {
                (param->u)[ j*np+(np-1) ]+= (param->heatsrcs[i].range-dist) / param->heatsrcs[i].range * param->heatsrcs[i].temp;
            }
        }
    }

    // Copy u into uhelp
    double *putmp = param->uhelp;
    double *pu = param->u;
    for( int k=0; k<np*np; k++ ) *putmp++ = *pu++;

    return 1;
}

int finalize( algoparam_t *param )
{
    if( param->u ) {
        if(PINNED) cudaFreeHost(param->u); else free(param->u);
        param->u = 0;
    }
    if( param->uhelp ) {
        if(PINNED) cudaFreeHost(param->uhelp); else free(param->uhelp);
        param->uhelp = 0;
    }
    if( param->uvis ) {
        if(PINNED) cudaFreeHost(param->uvis); else free(param->uvis);
        param->uvis = 0;
    }
    if( param->diffs ) { 
        if(PINNED) cudaFreeHost(param->diffs); else free(param->diffs); 
        param->diffs = 0; 
    }
    return 1;
}

void write_image( FILE * f, double *u, unsigned sizex, unsigned sizey )
{
    unsigned char r[1024], g[1024], b[1024];
    int i, j, k;
    double min, max;

    j=1023;
    for( i=0; i<256; i++ ) { r[j]=255; g[j]=i; b[j]=0; j--; }
    for( i=0; i<256; i++ ) { r[j]=255-i; g[j]=255; b[j]=0; j--; }
    for( i=0; i<256; i++ ) { r[j]=0; g[j]=255; b[j]=i; j--; }
    for( i=0; i<256; i++ ) { r[j]=0; g[j]=255-i; b[j]=255; j--; }

    min=DBL_MAX; max=-DBL_MAX;
    for( i=0; i<sizey; i++ ) {
        for( j=0; j<sizex; j++ ) {
            if( u[i*sizex+j]>max ) max=u[i*sizex+j];
            if( u[i*sizex+j]<min ) min=u[i*sizex+j];
        }
    }

    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", sizex, sizey);
    fprintf(f, "%u\n", 255);

    for( i=0; i<sizey; i++ ) {
        for( j=0; j<sizex; j++ ) {
            k=(int)(1023.0*(u[i*sizex+j]-min)/(max-min));
            fprintf(f, "%d %d %d  ", r[k], g[k], b[k]);
        }
        fprintf(f, "\n");
    }
}

int coarsen( double *uold, unsigned oldx, unsigned oldy , double *unew, unsigned newx, unsigned newy )
{
    int i, j, stepx, stepy;
    int stopx = newx;
    int stopy = newy;

    if (oldx>newx) stepx=oldx/newx; else { stepx=1; stopx=oldx; }
    if (oldy>newy) stepy=oldy/newy; else { stepy=1; stopy=oldy; }

    for( i=0; i<stopy-1; i++ ) {
        for( j=0; j<stopx-1; j++ ) {
            unew[i*newx+j]=uold[i*oldx*stepy+j*stepx];
        }
    }
    return 1;
}

#define BUFSIZE 100
int read_input( FILE *infile, algoparam_t *param )
{
    int i, n;
    char buf[BUFSIZE];
    fgets(buf, BUFSIZE, infile);
    n = sscanf(buf, "%u", &(param->numsrcs) );
    if( n!=1 ) return 0;
    (param->heatsrcs) = (heatsrc_t*) malloc( sizeof(heatsrc_t) * (param->numsrcs) );
    for( i=0; i<param->numsrcs; i++ ) {
        fgets(buf, BUFSIZE, infile);
        n = sscanf( buf, "%f %f %f %f", &(param->heatsrcs[i].posx), &(param->heatsrcs[i].posy), &(param->heatsrcs[i].range), &(param->heatsrcs[i].temp) );
        if( n!=4 ) return 0;
    }
    return 1;
}

void print_params( algoparam_t *param ) {
    int i;
    fprintf(stdout, "Iterations        : %u\n", param->maxiter);
    fprintf(stdout, "Resolution        : %u\n", param->resolution);
    fprintf(stdout, "Residual          : %f\n", param->residual);
    fprintf(stdout, "Num. Heat sources : %u\n", param->numsrcs);
    for( i=0; i<param->numsrcs; i++ ) {
        fprintf(stdout, "  %2d: (%2.2f, %2.2f) %2.2f %2.2f \n", i+1, param->heatsrcs[i].posx, param->heatsrcs[i].posy, param->heatsrcs[i].range, param->heatsrcs[i].temp );
    }
}

double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec+1e-6*tv.tv_usec;
}