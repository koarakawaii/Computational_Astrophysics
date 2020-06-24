#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <time.h>

void PRINT_MATRIX(int, double*);
void FPRINTF(FILE*, int, double, double *);
void INCOMPLETE_CHOLESKY(int, double*, double*);
void PRODUCE_CHOLESKY(int, int, double, double, double*);

int main(void)
{
	int N, row;
	double photon_mass, dx, start, end;
	double *A, *R, *R_produced;

	printf("Test the incomplete Cholesky factorization of discrete Laplacian matrix and reproduced it.\n\n");
	
	printf("Set the size of the lattice so the dimension (N,N) of the lattice so matrix is (N^2,N^2).\n");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d) .\n", N, N);
	printf("Set the value of photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("\n");
	
	FILE* output_matrix = fopen("Matrix_R_Incomplete_CPU.txt", "w");
	row = N*N;
	dx = 1./(N-1);
	A = (double*)calloc(row*row,sizeof(double));
	R = (double*)calloc(row*row,sizeof(double));
	R_produced = (double*)calloc(row*3,sizeof(double));
	srand(time(NULL));

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

	double* A_copy = malloc(row*row*sizeof(double));
	memcpy(A_copy, A, row*row*sizeof(double));

	printf("Start factorization...\n");
	start = clock();
	INCOMPLETE_CHOLESKY(row, A_copy, R);
	end = clock();
	printf("Done. Total factorization time is %.4f ms.\n",1000*(end-start)/CLOCKS_PER_SEC);

	printf("Start factorization with matrix...\n");
	start = clock();
	PRODUCE_CHOLESKY(N, row, dx, photon_mass, R_produced);
	end = clock();
	printf("Done. Total factorization time with out matrix is %.4f ms.\n",1000*(end-start)/CLOCKS_PER_SEC);

	FPRINTF(output_matrix, N*N, 1.0, R);
	
	output_matrix = fopen("Matrix_R_Incomplete_Without_Matrix_CPU.txt", "w");
	for (int i=0; i<row-N; i++)
	{
		for (int j=0; j<i; j++)
			fprintf(output_matrix, "%.4f\t", 0.0);
		fprintf(output_matrix, "%.4f\t", R_produced[3*i]);
		fprintf(output_matrix, "%.4f\t", R_produced[3*i+1]);
		for (int j=i+2; j<i+N; j++)
			fprintf(output_matrix, "%.4f\t", 0.0);
		fprintf(output_matrix, "%.4f\t", R_produced[3*i+2]);
		for (int j=i+N+1; j<row; j++)
			fprintf(output_matrix, "%.4f\t", 0.0);
		fprintf(output_matrix, "\n");
	}
	for (int i=row-N; i<row; i++)
	{
		for (int j=0; j<i; j++)
			fprintf(output_matrix, "%.4f\t", 0.0);
		fprintf(output_matrix, "%.4f\t", R_produced[3*i]);
		if (i!=row-1)
		{
			fprintf(output_matrix, "%.4f\t", R_produced[3*i+1]);
			for (int j=i+2; j<row; j++)
				fprintf(output_matrix, "%.4f\t", 0.0);
			fprintf(output_matrix, "\n");
		}
		else
			fprintf(output_matrix, "\n");
	}

	printf("Evaluate the error of with/without matrix factorization...\n");
	double error = 0.0;
	for (int i=0; i<row; i++)
	{
		int i_x = i%N;
		int i_y = i/N;
		error += pow(R_produced[3*i]-R[i*row+i],2.);
		if (i_x<N-2)
			error += pow(R_produced[3*i+1]-R[(i+1)*row+i],2.);
		if (i_y<N-2)
			error += pow(R_produced[3*i+2]-R[(i+N)*row+i],2.);
	}
	error = sqrt(error);
	printf("Done. Error is %.16e .\n", error);

	printf("Test the inverse of incomplete Cholesky factorization.\n");
	

	free(A);
	free(R);
	free(R_produced);
	fclose(output_matrix);
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
					A[j*row+i] = A[j*row+i] - R[j*row+k]*R[i*row+k];
		}
	}
	R[row*row-1] = sqrt(A[row*row-1]);
}

void PRODUCE_CHOLESKY(int N, int row, double dx, double photon_mass, double* R)
{
	int idx_x, idx_y;
	double temp;

	for (int idx=0; idx<N+1; idx++)
		R[3*idx] = 1.0;	//diagonal
	for (int idx=N+1; idx<row-N-1; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			R[3*idx] = (4.+pow(photon_mass*dx,2.));
			if (idx_x<N-2)
				R[3*idx+1] = -1.;
			if (idx_y<N-2)
				R[3*idx+2] = -1.;
		}
		else
			R[3*idx] = 1.0;
	}
	for (int idx=row-N-1; idx<row; idx++)
		R[3*idx] = 1.0;

	for (int idx=N+1; idx<row-N-1; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			temp = sqrt(R[3*idx]);
			R[3*idx] = temp;
//			printf("%.4f\n", R[3*idx]);
			if (idx_x<N-2)
			{
				R[3*idx+1] /= temp;
				R[3*idx+3] -= pow(R[3*idx+1],2.);
//				printf("%.4f\n", R[3*idx+1]);
			}
			if (idx_y<N-2)
			{
				R[3*idx+2] /= temp;
				R[3*(idx+N)] -= pow(R[3*idx+2],2.);
//				printf("%.4f\n", R[3*idx+2]);
			}
		}
//		else
//			printf("%.4f\n", R[3*idx]);
	}
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

