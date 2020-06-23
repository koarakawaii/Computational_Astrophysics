CUDIR     := /usr/local/cuda
GSLDIR	  := /home/lab/gsl

CC      := gcc
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 --compiler-options -fopenmp -I$(CUDIR)/include -I$(GSLDIR)/include -m64 -arch=compute_75 -code=sm_75 -Xptxas -v

BIN     := test_SSOR_M_MATRIX cholesky_factorization_CPU cholesky_factorization_GPU CONJUGATE_GRADIENT_SOLVER_TEST
FILE_M_TEST		:= test_SSOR_M_MATRIX.c 
FILE_CHOLESKY_CPU := cholesky_factorization_CPU.c
FILE_CHOLESKY_GPU := cholesky_factorization_GPU.cu
FILE_CG_TEST	:= CONJUGATE_GRADIENT_SOLVER_TEST.cu 
FILE_INCOMPLETE_CHOLESKY_CPU := incomplete_cholesky_factorization_CPU.c
LIB     := -lgsl -lgslcblas -llapack -lm -lcublas -lcurand -lcusolver

all: $(BIN)

M_TEST:	test_SSOR_M_MATRIX
test_SSOR_M_MATRIX:$(FILE_M_TEST)
	$(CC) $(CFLAGS) -o test_SSOR_M_MATRIX $(FILE_M_TEST) $(LIB)

CHOLESKY_CPU: cholesky_factorization_CPU
cholesky_factorization_CPU: $(FILE_CHOLESKY_CPU)
	$(CC) $(CFLAGS) -o cholesky_factorization_CPU $(FILE_CHOLESKY_CPU) $(LIB)

CHOLESKY_GPU: cholesky_factorization_GPU
cholesky_factorization_GPU: $(FILE_CHOLESKY_GPU)
	$(NVCC) -o cholesky_factorization_GPU $(NVFLAGS) $(FILE_CHOLESKY_GPU) $(LIB)

CG_TEST:test_CG
test_CG:$(FILE_CG_TEST)
	$(NVCC) -o CONJUGATE_GRADIENT_SOLVER_TEST $(NVFLAGS) $(FILE_CG_TEST) $(LIB)

INCOMPLETE_CHOLESKY_CPU: incomplete_cholesky_factorization_CPU
incomplete_cholesky_factorization_CPU: $(FILE_INCOMPLETE_CHOLESKY_CPU)
	$(CC) $(CFLAGS) -o incomplete_cholesky_factorization_CPU $(FILE_INCOMPLETE_CHOLESKY_CPU) $(LIB)

clean:
	rm -f $(BIN) 
