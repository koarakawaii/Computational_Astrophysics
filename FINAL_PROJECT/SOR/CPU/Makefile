MPICC      := mpicc
CFLAGS  := -O3 -Wall -g
MPICPP     := mpic++
CFLAGS  := -O3 -Wall -g

BIN     := ITERATION_SOLVER_SOR_MPI_ver_calculate 
FILE    := ITERATION_SOLVER_SOR_MPI_ver_calculate.c 
LIB     := -lm -fopenmp -lgslcblas

SOR: ITERATION_SOLVER_SOR_MPI_ver_calculate.c
	$(MPICC) -o $(BIN) $(CFLAGS) $(FILE) $(LIB)

clean:
	rm -f $(BIN) 
