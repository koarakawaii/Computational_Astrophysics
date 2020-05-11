CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v

BIN     := conjugate_gradient_non_Hermitian conjugate_gradient_Hermitian
FILE    := conjugate_gradient_non_Hermitian.cu conjugate_gradient_Hermitian.cu
LIB     := -lgsl -lgslcblas -lm -lcublas -lcurand

all: $(BIN)

non_hermitian: conjugate_gradient_non_Hermitian
hermitian: conjugate_gradient_Hermitian

conjugate_gradient_non_Hermitian: conjugate_gradient_non_Hermitian.cu
	$(NVCC) -o conjugate_gradient_non_Hermitian $(NVFLAGS) conjugate_gradient_non_Hermitian.cu $(LIB)
conjugate_gradient_Hermitian: conjugate_gradient_Hermitian.cu
	$(NVCC) -o conjugate_gradient_Hermitian $(NVFLAGS) conjugate_gradient_Hermitian.cu $(LIB)

clean:
	rm -f $(BIN) 
