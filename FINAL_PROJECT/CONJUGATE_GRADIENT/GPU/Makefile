CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 --compiler-options -fopenmp -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v

BIN     := CONJUGATE_GRADIENT_SOLVER
FILE    := CONJUGATE_GRADIENT_SOLVER.cu 
LIB     := -lgsl -lgslcblas -lm -lcublas -lcurand

all: $(BIN)

CG: CONJUGATE_GRADIENT_SOLVER

CONJUGATE_GRADIENT_SOLVER: CONJUGATE_GRADIENT_SOLVER.cu
	$(NVCC) -o CONJUGATE_GRADIENT_SOLVER $(NVFLAGS) CONJUGATE_GRADIENT_SOLVER.cu $(LIB)

clean:
	rm -f $(BIN) 
