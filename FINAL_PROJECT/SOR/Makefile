CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 --compiler-options -fopenmp -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v

BIN     := ITERATION_SOLVER_SOR_ver_allocate ITERATION_SOLVER_SOR_ver_calculate
FILE    := ITERATION_SOLVER_SOR_ver_allocate.cu ITERATION_SOLVER_SOR_ver_calculate.cu
LIB     := -lgsl -lgslcblas -lm -lcublas -lcurand

all: $(BIN)

SOR_a: ITERATION_SOLVER_SOR_ver_allocate
SOR_c: ITERATION_SOLVER_SOR_ver_calculate

ITERATION_SOLVER_SOR_ver_allocate: ITERATION_SOLVER_SOR_ver_allocate.cu
	$(NVCC) -o ITERATION_SOLVER_SOR_ver_allocate $(NVFLAGS) ITERATION_SOLVER_SOR_ver_allocate.cu $(LIB)

ITERATION_SOLVER_SOR_ver_calculate: ITERATION_SOLVER_SOR_ver_calculate.cu
	$(NVCC) -o ITERATION_SOLVER_SOR_ver_calculate $(NVFLAGS) ITERATION_SOLVER_SOR_ver_calculate.cu $(LIB)

clean:
	rm -f $(BIN) 
