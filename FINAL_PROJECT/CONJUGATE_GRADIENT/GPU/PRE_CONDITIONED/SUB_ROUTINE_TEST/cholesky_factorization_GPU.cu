#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void PRINT_MATRIX(int, double*);
void FPRINTF(FILE*, int, double, double *);
/* To link to lapack, need to declare a extern function like this. */

__global__ void FILL_MATRIX(int N, double dx, double photon_mass, double* A)
{
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N;
	int row = N*N;

	if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
	{
		if (idx_x>1)
            A[(idx-1)*row + idx] = -1.;
        if (idx_x<N-2)
            A[(idx+1)*row + idx] = -1.;
        if (idx_y>1)
            A[(idx-N)*row + idx] = -1.;
        if (idx_y<N-2)
            A[(idx+N)*row + idx] = -1.;
        A[idx*row + idx] = 4. - pow(photon_mass*dx,2.);
	}
	else
        A[idx*row + idx] = 1.;
}

int main(void)
{
	int N, row, tx, ty, bx, by;
	float gpu_time;
	double photon_mass, dx;
	double *A, *A_copy;
	cudaEvent_t start, stop;

	printf("Test the Cholesky factorization of discrete Laplacian matrix using cusolver.\n\n");
	
	printf("Set the size of the lattice so the dimension (N,N) of the lattice so matrix is (N^2,N^2).\n");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d) .\n", N, N);
	row = N*N;
	printf("Set the value of photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("Set the threads per block (tx,ty) (N must be divisible by both tx and ty)");
	scanf("%d %d",&tx, &ty);
	if (N%tx!=0)
	{
		printf("N is not divisible by tx! Exit!");
		return EXIT_FAILURE;
	}
	if (N%ty!=0)
	{
		printf("N is not divisible by ty! Exit!");
		return EXIT_FAILURE;
	}
	printf("Threads per block is (%d,%d) .\n", tx, ty);
	printf("Blocks per grid will be set automatically.\n");
	bx = N/tx;
	by = N/ty;
	printf("Blocks per grid is (%d,%d) .\n", bx, by);
	printf("\n");
	
	dx = 1./(N-1);
	cudaSetDevice(0);
	dim3 bpg(bx, by);
	dim3 tpb(tx, ty);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cusolverDnHandle_t solver_handle = NULL;
	cusolverDnCreate(&solver_handle);
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
	int work_size;
	int* dev_info;
	cudaMalloc(&dev_info, sizeof(int));
	
	cudaMallocManaged(&A, row*row*sizeof(double));
	cudaMallocManaged(&A_copy, row*row*sizeof(double));
	cudaMemset(A, 0, row*row*sizeof(double));
	FILL_MATRIX<<<bpg,tpb>>>(N, dx, photon_mass, A);
	cudaDeviceSynchronize();

	cudaMemcpy(A_copy, A, row*row*sizeof(double), cudaMemcpyDeviceToDevice);
	cusolverDnDpotrf_bufferSize(solver_handle, uplo, row, A_copy, row, &work_size);
	double* work;
	cudaMalloc(&work, work_size*sizeof(double));

	printf("Start factorization...\n");
	cudaEventRecord(start,0);
	if (cusolverDnDpotrf(solver_handle, uplo, row, A_copy, row, work, work_size, dev_info)==CUSOLVER_STATUS_SUCCESS)
	{
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		printf("Done. Factorization succeed! Total factorization time is %.4f ms.\n",gpu_time);
//		PRINT_MATRIX(row, A_copy);
	}
	else
	{
		printf("Factorization failed!\n");
		return EXIT_FAILURE;
	}

	printf("Set the lower triangular part as 0...\n");
	/* set the lower triangular part as 0 */
	for (int j=0; j<row; j++)
	{
		for (int i=j+1; i<row; i++)
			A_copy[i+j*row] = 0.0;
	}
	printf("Done.\n");

	FILE* output = fopen("Matrix_R_GPU.txt", "w");
	FPRINTF(output, row, 1.0, A_copy);
	fclose(output);

//	PRINT_MATRIX(row, A);

	double* A_prime = (double *)malloc(row*row*sizeof(double));
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

	cudaFree(A);
	cudaFree(A_copy);
	cudaFree(A_prime);
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
            fprintf(output_file, "%.4f\t", scale*A[j*row+i]);
        fprintf(output_file, "\n");
    }
}

