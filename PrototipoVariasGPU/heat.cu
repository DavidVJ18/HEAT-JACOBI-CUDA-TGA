#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "heat.h"

void usage( char *s ) {
    fprintf(stderr, "Usage: %s input_file [-n num_iter -s resolution -a solver -u extra -o result_file]\n", s);
    fprintf(stderr, "       input_file with heat sources\n");
    fprintf(stderr, "       -n to specify the number of iterations to stop iterative algorithm (0 no stop, 25000 default)\n");
    fprintf(stderr, "       -s to specify the resolution (size) of the squared hot surface (254 default)\n");
    fprintf(stderr, "       -r to specify the residual value to stop iterative algorithm (0.00005 default)\n");
    fprintf(stderr, "       -a to specify the solver to use. 0: Jacobi; 1: Gauss-Seidel (default 0)\n");
    fprintf(stderr, "       -o to specify the name of the output file with heat map generated (default heat.ppm)\n");
    fprintf(stderr, "       -u extra argument to be used by the programmer\n");
}

int main( int argc, char *argv[] ) {
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename = "heat.ppm";
    algoparam_t param;
    double runtime, flop;
    double residual=0.0;

    // Configuración por defecto
    param.maxiter = 25000;
    param.resolution = 254;
    param.visres = 254; // <--- INICIALIZACIÓN POR DEFECTO AÑADIDA
    param.residual = 0.00005;
    param.algorithm = 0; // Jacobi

    // Procesar argumentos
    if( argc < 2 ) { usage(argv[0]); return EXIT_FAILURE; }
    if( !(infile=fopen(argv[1], "r"))  ) { fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]); usage(argv[0]); return EXIT_FAILURE; }
    
    for (int i=2; i<argc-1; i++) {
        if (strcmp(argv[i], "-n")==0) param.maxiter = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s")==0) param.resolution = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r")==0) param.residual = atof(argv[++i]);
        else if (strcmp(argv[i], "-o")==0) resfilename = argv[++i];
    }
    
    // <--- CORRECCIÓN IMPORTANTE:
    // Aseguramos que la resolución de la imagen coincida con la de simulación
    // o con lo que hayas pasado por parámetro.
    param.visres = param.resolution; 

    read_input(infile, &param);
    print_params(&param);

    // Inicializar Host Memory
    if( !initialize(&param) ) return EXIT_FAILURE;

    // --- SETUP MULTI-GPU ---
    int numDevs = 0;
    cudaGetDeviceCount(&numDevs);
    if (numDevs > MAX_GPUS) numDevs = MAX_GPUS;
    param.numDevs = numDevs;
    
    printf("Ejecutando con %d GPUs\n", numDevs);

    // Habilitar P2P (Peer Access)
    for(int i=0; i<numDevs; i++) {
        cudaSetDevice(i);
        for(int j=0; j<numDevs; j++) {
            if (i != j) {
                int canAccess = 0;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                if (canAccess) cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // Calcular particionamiento
    int np = param.resolution + 2; // Ancho y alto total
    int rows_per_gpu = param.resolution / numDevs; // Filas de trabajo (sin halos)
    int remainder = param.resolution % numDevs;

    int current_row_start = 1; // Empezamos en la fila 1 (la 0 es borde global)

    for (int i = 0; i < numDevs; i++) {
        cudaSetDevice(i);

        // Calcular altura local para esta GPU (filas trabajo + 2 halos)
        int my_rows = rows_per_gpu + (i < remainder ? 1 : 0);
        param.local_sizex[i] = my_rows + 2; 
        param.local_sizey = np;

        size_t size_bytes = param.local_sizex[i] * np * sizeof(double);
        
        // Mallocs
        cudaMalloc((void**)&param.u_devs[i], size_bytes);
        cudaMalloc((void**)&param.uhelp_devs[i], size_bytes);
        cudaMalloc((void**)&param.diffsBlock_devs[i], 1024 * 1024 * sizeof(double)); 
        cudaMalloc((void**)&param.partial_sums_devs[i], 1024 * sizeof(double));

        // COPIA H2D: Host -> Device (incluyendo halos)
        int host_offset_rows = current_row_start - 1;
        
        cudaMemcpy(
            param.u_devs[i], 
            param.u + host_offset_rows * np, 
            size_bytes, 
            cudaMemcpyHostToDevice
        );
        
        cudaMemcpy(param.uhelp_devs[i], param.u_devs[i], size_bytes, cudaMemcpyDeviceToDevice);

        current_row_start += my_rows;
    }

    // --- BUCLE PRINCIPAL ---
    runtime = wtime();
    iter = 0;
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);
    cudaEventRecord(kernelStart, 0); cudaEventSynchronize(kernelStart);
    while(1) {
        residual = solve_cuda_multi(&param);
        swap_mats_dev_multi(&param);
        halo_exchange_multi(&param);

        iter++;
        if (residual < param.residual) break;
        if (param.maxiter > 0 && iter >= param.maxiter) break;
    }
    cudaEventRecord(kernelStop, 0); cudaEventSynchronize(kernelStop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
    runtime = wtime() - runtime;

    // --- RECOGER RESULTADOS (D2H) ---
    current_row_start = 1;
    for (int i = 0; i < numDevs; i++) {
        cudaSetDevice(i);
        
        // Copiamos solo la parte válida (sin halos externos)
        int valid_rows = param.local_sizex[i] - 2;
        int size_bytes_valid = valid_rows * np * sizeof(double);
        
        cudaMemcpy(
            param.u + current_row_start * np, 
            param.u_devs[i] + 1 * np, 
            size_bytes_valid, 
            cudaMemcpyDeviceToHost
        );
        
        current_row_start += valid_rows;
        
        cudaFree(param.u_devs[i]);
        cudaFree(param.uhelp_devs[i]);
        cudaFree(param.diffsBlock_devs[i]);
        cudaFree(param.partial_sums_devs[i]);
    }
    
    fprintf(stdout, "Time: %04.3f \n", kernelTime);
    flop = iter * 11.0 * param.resolution * param.resolution;
    fprintf(stdout, "Flops: %3.3f GFlop => %6.2f MFlop/s\n", flop/1e9, flop/runtime/1e6);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    if (resfile=fopen(resfilename, "w")) {
        // Aquí usa param.visres, que ahora sí tiene el valor correcto
        coarsen(param.u, np, np, param.uvis, param.visres+2, param.visres+2);
        write_image(resfile, param.uvis, param.visres+2, param.visres+2);
        fclose(resfile);
    }
    
    finalize(&param);
    return 0;
}
