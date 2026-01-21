CUDA_HOME   = /Soft/cuda/12.2.2
NVCC        = $(CUDA_HOME)/bin/nvcc
# Arquitecturas para Boada (RTX 3080 y 4090)
ARCH        = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89

# Flags: O3 (optimización), ptxas verbose (info recursos)
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64

EXE	        = heat.exe
OBJ	        = heat.o solver.o misc.o

default: $(EXE)

# Regla general: Cualquier .o se crea desde su .cu
%.o: %.cu heat.h
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

# Enlazar todo
$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) *.ppm submit*