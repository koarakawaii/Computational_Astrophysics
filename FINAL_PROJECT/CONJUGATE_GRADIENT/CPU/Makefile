MPICC      := mpicc
CC		   := gcc
CFLAGS  := -O3 -Wall -g
MPICPP     := mpic++

BIN     := CONJUGATE_GRADIENT_SOLVER_MPI_BLAS CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS 
FILE_BLAS    := CONJUGATE_GRADIENT_SOLVER_MPI_BLAS.c
FILE_NO_BLAS := CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS.c
FILE_OMP	 := CONJUGATE_GRADIENT_SOLVER_OPENMP.c
LIB     := -lm -lgslcblas -fopenmp

all: $(BIN)

CG_BLAS: CONJUGATE_GRADIENT_SOLVER_MPI_BLAS
CONJUGATE_GRADIENT_SOLVER_MPI_BLAS: CONJUGATE_GRADIENT_SOLVER_MPI_BLAS.c
	$(MPICC) -o CONJUGATE_GRADIENT_SOLVER_MPI_BLAS $(CFLAGS) $(FILE_BLAS) $(LIB)

CG_NO_BLAS: CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS
CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS: CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS.c
	$(MPICC) -o CONJUGATE_GRADIENT_SOLVER_MPI_without_BLAS $(CFLAGS) $(FILE_NO_BLAS) $(LIB)

CG_OMP: CONJUGATE_GRADIENT_SOLVER_OPENMP
CONJUGATE_GRADIENT_SOLVER_OPENMP: CONJUGATE_GRADIENT_SOLVER_OPENMP.c
	$(CC) -o CONJUGATE_GRADIENT_SOLVER_OPENMP $(CFLAGS) $(FILE_OMP) $(LIB)

clean:
	rm -f $(BIN) 
