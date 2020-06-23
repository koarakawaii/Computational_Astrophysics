#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void PRINT_MATRIX(int, double*);
void FPRINTF(FILE*, int, double, double *);
void INCOMPLETE_CHOLESKY(int, double*, double*);

int main(void)
{
	int N, row;
	double photon_mass, dx, start, end;
	double *A, *R;

	printf("Test the Cholesky factorization of discrete Laplacian matrix using LAPACK.\n\n");
	
	printf("Set the size of the lattice so the dimension (N,N) of the lattice so matrix is (N^2,N^2).\n");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d) .\n", N, N);
	printf("Set the value of photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("\n");
	
	row = N*N;
	dx = 1./(N-1);
	A = (double*)calloc(row*row,sizeof(double));
	R = (double*)calloc(row*row,sizeof(double));

	for (int i=0; i<N*N; i++)
	{
		int i_x = i%N;
		int i_y = i/N;
		
		if (i_x!=0&&i_x!=N-1&&i_y!=0&&i_y!=N-1)
		{
			if (i_x>1)
				A[(i-1)*row + i] = -1.;
			if (i_x<N-2)
				A[(i+1)*row + i] = -1.;
			if (i_y>1)
				A[(i-N)*row + i] = -1.;
			if (i_y<N-2)
				A[(i+N)*row + i] = -1.;
			A[i*row + i] = 4. - pow(photon_mass*dx,2.);
		}
		else
			A[i*row + i] = 1.;
	}

	printf("Start factorization...\n");
	start = clock();
	INCOMPLETE_CHOLESKY(row, A, R);

	end = clock();
	printf("Done. Factorization succeed! Total factorization time is %.4f ms.\n",1000*(end-start)/CLOCKS_PER_SEC);

	FILE* output = fopen("Matrix_R_Incomplete_CPU.txt", "w");
	FPRINTF(output, N*N, 1.0, R);
	fclose(output);

	free(A);
	free(R);
	return EXIT_SUCCESS;
}

void INCOMPLETE_CHOLESKY(int row, double* A, double* R)
{
	for (int k=0; k<row-1; k++)
	{
		R[k*row+k] = sqrt(A[k*row+k]);
		for (int j=k+1; j<row; j++)
			R[j*row+k] = A[j*row+k]/R[k*row+k];
		for (int i=k+1; i<row; i++)
		{
			for (int j=i; j<row; j++)
				if (A[j*row+i]!=0)
					A[j*row+i] = A[j*row+i] - R[j*row+k]*R[j*row+k];
					
		}
	}
	R[row*row-1] = sqrt(A[row*row-1]);
}

void PRINT_MATRIX(int row, double* A)
{
	for (int i=0; i<row; i++)
	{
		for (int j=0; j<row; j++)
			printf("%.4f\t", A[j*row + i]);
		printf("\n");
	}
	printf("\n");
}

void FPRINTF(FILE *output_file, int N, double scale, double *array)
{
    for (int j=0; j<N; j++)
    {
        for (int i=0; i<N; i++)
            fprintf(output_file, "%.4f\t", scale*array[i+j*N]);
        fprintf(output_file, "\n");
    }
}

