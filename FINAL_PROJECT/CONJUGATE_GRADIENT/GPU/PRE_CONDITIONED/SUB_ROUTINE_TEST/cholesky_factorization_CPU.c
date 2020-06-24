#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>

void PRINT_MATRIX(int, double*);
void FPRINTF(FILE*, int, double, double *);
/* To link to lapack, need to declare a extern function like this. */
extern void  dpotrf_(char*, int*, double*, int*, int*); 

int main(void)
{
	char handle = 'U';
	int N, row, status;
	double photon_mass, dx, start, end;
	double *A, *A_copy;

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
	A_copy = (double*)malloc(row*row*sizeof(double));

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

	memcpy(A_copy, A, row*row*sizeof(double));
//	PRINT_MATRIX(row, A);

	printf("Start factorization...\n");
	start = clock();
	/* use the lapack function */
	dpotrf_(&handle, &row, A_copy, &row, &status);

	if (status==0)
	{
		end = clock();
		printf("Done. Factorization succeed! Total factorization time is %.4f ms.\n",1000*(end-start)/CLOCKS_PER_SEC);
		end = clock();
	}
	else
		printf("Factorization failed! Status is %d .\n", status);

	printf("Set the lower triangular part as 0...\n");
	/* set the lower triangular part as 0 */
	for (int j=0; j<row; j++)
	{
		for (int i=j+1; i<row; i++)
			A_copy[i+j*row] = 0.0;
	}
	printf("Done.\n");
//	PRINT_MATRIX(row, A);

	FILE* output = fopen("Matrix_R_CPU.txt", "w");
	FPRINTF(output, N*N, 1.0, A_copy);
	fclose(output);

	double* A_prime = malloc(row*row*sizeof(double));
	double temp;

	printf("Check the multiplication...\n");
	for (int j=0; j<row; j++)
	{
		for (int i=j; i<row; i++)
		{
			temp = cblas_ddot(row, A_copy+i*row, 1, A_copy+j*row, 1);
			A_prime[i*row+j] = temp;
//			printf("%d\t%d\t%.4f\n", j, i, temp);
			if (i!=j)
				A_prime[j*row+i] = temp;
		}
	}
	printf("Done.\n");
//	PRINT_MATRIX(row, A_prime);
	printf("Evaluate error...\n");
	double error = 0.0;
	for (int i=0; i<row*row; i++)
		error += pow(A[i]-A_prime[i],2.);
	printf("Done. Error is %.16e .\n", sqrt(error));

	free(A);
	free(A_copy);
	free(A_prime);
	return EXIT_SUCCESS;
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

void FPRINTF(FILE *output_file, int row, double scale, double *A)
{
    for (int i=0; i<row; i++)
    {
        for (int j=0; j<row; j++)
            fprintf(output_file, "%.4f\t", scale*A[i+j*row]);
        fprintf(output_file, "\n");
    }
}

