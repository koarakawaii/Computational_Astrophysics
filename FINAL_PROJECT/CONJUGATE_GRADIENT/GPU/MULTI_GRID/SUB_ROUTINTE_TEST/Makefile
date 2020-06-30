CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 --compiler-options -fopenmp -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v

BIN     := GRIDDING inverse_LU_GPU
FILE_GRID    := GRIDDING.cu 
FILE_LU	:= inverse_LU_GPU.cu
FILE_TGCG_proto    := CONJUGATE_GRADIENT_SOLVER_TWO_GRID_proto.cu 
LIB     := -lgsl -lgslcblas -lm -lcublas -lcurand

all: $(BIN)

GRID: GRIDDING
GRIDDING: GRIDDING.cu
	$(NVCC) -o GRIDDING $(NVFLAGS) $(FILE_GRID) $(LIB)

LU: inverse_LU_GPU
inverse_LU_GPU: inverse_LU_GPU.cu
	$(NVCC) -o inverse_LU_GPU $(NVFLAGS) $(FILE_LU) $(LIB)

TGCG_proto: CONJUGATE_GRADIENT_SOLVER_TWO_GRID_proto
CONJUGATE_GRADIENT_SOLVER_TWO_GRID_proto: CONJUGATE_GRADIENT_SOLVER_TWO_GRID_proto.cu
	$(NVCC) -o CONJUGATE_GRADIENT_SOLVER_TWO_GRID_proto $(NVFLAGS) $(FILE_TGCG_proto) $(LIB)

clean:
	rm -f $(BIN) 
