MPICC      := mpicc
CFLAGS  := -O3 -Wall -g
MPICPP     := mpic++
CFLAGS  := -O3 -Wall -g

BIN     := CONJUGATE_GRADIENT_SOLVER_MPI 
FILE    := CONJUGATE_GRADIENT_SOLVER_MPI.c 
LIB     := -lm -lgslcblas -fopenmp

SOR: CONJUGATE_GRADIENT_SOLVER_MPI.c
	$(MPICC) -o $(BIN) $(CFLAGS) $(FILE) $(LIB)

clean:
	rm -f $(BIN) 
