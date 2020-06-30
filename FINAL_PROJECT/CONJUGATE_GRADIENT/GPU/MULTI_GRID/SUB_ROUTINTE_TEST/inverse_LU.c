#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

double DOT(int, double*, double*);
void MATRIX_MULTIPLY_LU(int, int, double**, double**);
void INVERSE_MATRIX(int, double*, double*);

int main(void)
{
	int N, row;
    double photon_mass, dx, start, end;
    double *A, *A_inverse;
	
	puts("Do LU-decomposition for matrix A .\n\n");
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
    A_inverse = (double*)malloc(row*row*sizeof(double));

    for (int i=0; i<N*N; i++)
    {
        int i_x = i%N;
        int i_y = i/N;

        if (i_x!=0&&i_x!=N-1&&i_y!=0&&i_y!=N-1)
        {
            if (i_x>1)
                A[(i-1) + i*row] = -1.;
            if (i_x<N-2)
                A[(i+1) + i*row] = -1.;
            if (i_y>1)
                A[(i-N) + i*row] = -1.;
            if (i_y<N-2)
                A[(i+N) + i*row] = -1.;
            A[i*(row + 1)] = 4. - pow(photon_mass*dx,2.);
        }
        else
            A[i*(row + 1)] = 1.;
    }

	start = clock();
	INVERSE_MATRIX(row, A, A_inverse);
	end = clock();
	
//	puts("\nPrint out A matrix:");
//	for (int i=0; i<N_row; i++)
//	{
//		for (int j=0; j<N_col; j++)
//			printf("%.6f\t", A[i*N_col+j]);
//		printf("\n");
//	}
//	
//	puts("\nPrint out A_inverse matrix:");
//	for (int i=0; i<N_row; i++)
//	{
//		for (int j=0; j<N_col; j++)
//			printf("%.6f\t", A_inverse[j*N_col+i]);
//		printf("\n");
//	}
	puts("\nPrint out the error of A*A_inverse:");
	double **SUM = (double**)malloc(row*sizeof(double*));
	double error = 0.0;
	for (int i=0; i<row; i++)
	{
		SUM[i] = (double*)calloc(row, sizeof(double));
		for (int j=0; j<row; j++)
		{
			for (int k=0; k<row; k++)
				SUM[i][j] += A[i*row+k]*A_inverse[j*row+k];
//			printf("%.6f\t", SUM[i][j]);
			if (i==j)
				error += pow(SUM[i][i]-1.0,2.);
			else
				error += pow(SUM[i][j],2.);
		}
//		printf("\n");
	}
	printf("Error = %.16e , total calculation time is %.2f ms.\n", sqrt(error)/row, 1000*(end-start)/CLOCKS_PER_SEC);

	free(A);
	free(A_inverse);
	return(0);
}

double DOT(int dim, double *a, double *b)
{
	double sum = 0.;
	for (int i=0; i<dim; i++)
		sum += a[i]*b[i];
	return sum;
}

/*
Here U is already transported, so the matirx multiply is a little bit different from conventional multiply.
*/
void MATRIX_MULTIPLY_LU(int N_row, int N_col, double **L, double **U)
{
	double **SUM = (double**)malloc(N_row*sizeof(double*));
	for (int i=0; i<N_row; i++)
	{
		SUM[i] = (double*)calloc(N_col, sizeof(double));
		for (int j=0; j<N_col; j++)
		{
			for (int k=0; k<=i; k++)
			{
				if (k<=j)
				// Here is the difference!!
					SUM[i][j] += L[i][k]*U[j][k];
				//
			}
			printf("%.6f\t", SUM[i][j]);
		}
		printf("\n");
	}
	free(SUM);
}

void INVERSE_MATRIX(int row, double* A, double* A_inverse)
{
	int N_col = row;
	int N_row = row;

	printf("A is a %d by %d matrix.\n", N_row, N_col);
	puts("\nStart LU-decomposition...");
	int pivot_index;
	double temp_dot;
	double* temp_L, *temp_U;
	double* temp_A = (double*)malloc(N_col*sizeof(double));
	double** L = (double**)malloc(N_row*sizeof(double*));
	double** U = (double**)malloc(N_col*sizeof(double*));
	for (int i=1; i<=N_row; i++)
		L[i-1] = (double*)malloc(i*sizeof(double));
	for (int i=1; i<=N_col; i++)
	{
		U[i-1] = (double*)malloc(i*sizeof(double));
		U[i-1][i-1] = 1.;
	}
	//

	pivot_index = 0;
	L[0][0] = A[0];
	while (L[0][0]==0.)
	{
		memcpy(temp_A, A+(pivot_index+1)*N_col, N_col*sizeof(double));
		memcpy(A+(pivot_index+1)*N_col, A, N_col*sizeof(double));
		memcpy(A, temp_A, N_col*sizeof(double));
		pivot_index ++;
		printf("\tIndex %d row exchange with index %d row.\n", 0, pivot_index);
		L[0][0] = A[0];
	}

	for (int i=1; i<N_row; i++)
		L[i][0] = A[i*N_col];
	for (int i=1; i<N_col; i++)
		U[i][0] = A[i]/L[0][0];

	for (int j=1; j<N_col; j++)
	{
		pivot_index = j;
		temp_L = malloc(j*sizeof(double));
		L[j][j] = A[j*(N_col+1)] - DOT(j, L[j], U[j]);
		while (L[j][j]==0.&&pivot_index+1<N_row)
		{
			memcpy(temp_A, A+(pivot_index+1)*N_col, N_col*sizeof(double));
			memcpy(temp_L, L+(pivot_index+1)*N_col, j*sizeof(double));
			memcpy(A+(pivot_index+1)*N_col, A+j*N_col, N_col*sizeof(double));
			memcpy(L[pivot_index+1], L[j], j*sizeof(double));
			memcpy(A+j*N_col, temp_A, N_col*sizeof(double));
			memcpy(L+j*N_col, temp_L, j*sizeof(double));
			pivot_index ++;
			printf("\tIndex %d row exchange with index %d row.\n", j, pivot_index);
			L[j][j] = A[j*(N_col+1)] - DOT(j, L[j], U[j]);
		}
		
		for (int i=j+1; i<N_row; i++)
			L[i][j] = A[i*N_col+j] - DOT(j, L[i], U[j]);
		for (int i=j+1; i<N_col; i++)
			U[i][j] = (A[j*N_col+i] - DOT(j, L[j], U[i]))/L[j][j];
	}
	puts("\nLU-decomposition ends...");
	puts("\nCalculate the inverse matrix of A.");
	double *b;
	for (int j=0; j<N_col; j++)
	{
		b = (double*)calloc(N_row, sizeof(double));
		b[j] = 1.;
		A_inverse[j*N_col] = b[0]/L[0][0];
		for (int i=1; i<N_row; i++)
			A_inverse[j*N_col+i] = (b[i] - DOT(i, L[i], A_inverse+j*N_col))/L[i][i];
		for (int i=N_row-2; i>=0; i--)
		{
			int count = N_row-1-i;
			temp_U = (double*)malloc(count*sizeof(double));
			for (int k=0; k<count; k++)
				temp_U[k] = U[i+1+k][i];
			A_inverse[j*N_col+i] -= DOT(count, temp_U, A_inverse+j*N_col+i+1);
		}
	}

	free(L);
	free(U);
	free(temp_A);
	free(temp_L);
	free(temp_U);
}
