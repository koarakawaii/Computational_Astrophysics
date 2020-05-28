/* Matrices are represented in column-major format. */
/* Matrix product calcualted by cublasgemm seems not precise enough, so A^HA is computed by self-defined function.*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#define t_B 32					// threads per block
#define b_G 1024				// blocks per grid
#define amplitude 10.0			// scale the matrix and vector elements
#define IDX2C(i,j,ld) (j*ld+i)	// mapping of memory location

#define USE_CURAND false		// use the curand library to generate A and |b> randomly.
#define DO_CPU_CHECK true		// Do CG on CPU

void MHM_PRODUCT(int, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *);

#if (USE_CURAND==true)
int dimension;
__global__ void SET_RNG(long seed, curandState *state)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(seed, index, 0, &state[index]);
}

__global__ void FILLING_WITH_RNG(int N, cuDoubleComplex *A, curandState *state)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int shift = blockDim.x*gridDim.x;
	curandState local = state[index];
	while (index<N)
	{
		A[index].x = amplitude*curand_uniform(&local);
		A[index].y = amplitude*curand_uniform(&local);
		index += shift;
	}
	state[threadIdx.x + blockIdx.x*blockDim.x] = local;
}
#else
#define dimension 5	
#endif

#if (DO_CPU_CHECK==true)
void VV_ADDITION(int, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *);
cuDoubleComplex VV_PRODUCT(int, cuDoubleComplex *, cuDoubleComplex *);
void MV_PRODUCT(char, int, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *);
#endif

int main(void)
{
	puts("Solve the linear system A|x>=|b> by conjugate gradient on GPU.\n");
#if (USE_CURAND==true)
	puts("Set the dimension for the vector space.");
	scanf("%d", &dimension);
	printf("The dimension for the vector is %d .\n\n", dimension);
#endif
	char print_out;
	int iter = 0;
	float time_GPU, time_CPU;
	double error, criteria, norm;
	cuDoubleComplex alpha, beta, lambda_k, u_k;
	cuDoubleComplex *A, *b, *temp_matrix;
	cuDoubleComplex *A_dev, *b_dev, *x_k_dev, *r_k_dev, *p_k_dev;
	cuDoubleComplex *x_k_host = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex)*dimension);
	cudaEvent_t start, stop;

	A = (cuDoubleComplex *)malloc(dimension*dimension*sizeof(cuDoubleComplex));
	temp_matrix = (cuDoubleComplex *)malloc(dimension*dimension*sizeof(cuDoubleComplex));
	b = (cuDoubleComplex *)malloc(dimension*sizeof(cuDoubleComplex));
	cudaMalloc((void **)&A_dev, sizeof(cuDoubleComplex)*dimension*dimension);
	cudaMalloc((void **)&b_dev, sizeof(cuDoubleComplex)*dimension);
	cudaMalloc((void **)&x_k_dev, sizeof(cuDoubleComplex)*dimension);
	cudaMalloc((void **)&r_k_dev, sizeof(cuDoubleComplex)*dimension);
	cudaMalloc((void **)&p_k_dev, sizeof(cuDoubleComplex)*dimension);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

#if (USE_CURAND==true)
	long seed = 684128862;
	curandState *state_dev;
	cudaMalloc((void **)&state_dev, sizeof(curandState)*t_B*b_G);
	SET_RNG<<<b_G, t_B>>>(seed, state_dev);
	FILLING_WITH_RNG<<<b_G, t_B>>>(dimension*dimension, A_dev, state_dev);
	FILLING_WITH_RNG<<<b_G, t_B>>>(dimension, b_dev, state_dev);
	cublasGetMatrix(dimension, dimension, sizeof(cuDoubleComplex), A_dev, dimension, A, dimension);
	cublasGetVector(dimension, sizeof(cuDoubleComplex), b_dev, 1, b, 1);
	cudaFree(state_dev);
	scanf("%c", &print_out);
#else
	int index;
	double A_real[dimension][dimension] = { {1.0, 0.5, 0.0, 0.2645, 0.0}, 
											{1.0, 0.33, 2.123, 0.0, 0.001},
											{0.0, 0.0, 0.215, 0.0, 0.0},
											{0.249, 0.0, 0.0131, 0.013, 1.0},
											{0.0, 0.123, 0.0127, 0.0, 0.011} };
	double A_imag[dimension][dimension] = { {0., 0.0, 0.0, 0.0, 0.0}, 
											{0.99, 0.0, 1.0, 0.0, 0.0},
											{0.0, 1.0, 0.0, 0.0, 0.0},
											{0.0, 0.0, 0.0, 0.0, -8.97},
											{0.0, 0.0, 0.0, 0.0, 0.0} };
	
	double b_real[dimension] = {1.001, 1.0, 0.0 ,0.0, 0.0};
	double b_imag[dimension] = {0.0, 0.877, 0.0 ,0.0, 0.0};

	for (int j=0; j<dimension; j++)
	{
		b[j] = make_cuDoubleComplex(b_real[j], b_imag[j]);
		x_k_host[j] = make_cuDoubleComplex(0.0, 0.0);
		for (int i=0; i<dimension; i++)
		{
			index = IDX2C(i,j,dimension);
			A[index] = make_cuDoubleComplex(A_real[j][i], A_imag[j][i]);
		}
	}
#endif
	
	cublasMath_t mode = CUBLAS_TENSOR_OP_MATH;
	cublasPointerMode_t mode_pt = CUBLAS_POINTER_MODE_HOST;
	cublasHandle_t handle;

	puts("Print out the matrix A nd vector |b>? (y/n)");
	scanf("%c", &print_out);
	if (print_out=='y')
	{
		puts("Matrix A is:");
		for (int j=0; j<dimension; j++)
		{
			for (int i=0; i<dimension; i++)
				printf("%.6f%+.6fI\t", A[IDX2C(j,i,dimension)].x, A[IDX2C(j,i,dimension)].y);
			printf("\n");
		}
		printf("\n");
		puts("vector |b> is:");
		for (int i=0; i<dimension; i++)
			printf("%.6f%+.6fI\n", b[i].x, b[i].y);
		printf("\n");
	}
	else if (print_out!='n')
	{
		puts("Wrong input! Exit!");
		EXIT_FAILURE;
	}
	
	puts("Set the stopping criteria.");
	scanf("%lf", &criteria);
	printf("The stopping criteria is %.4e .\n", criteria);
	
	// generate A^HA
	MHM_PRODUCT(dimension, A, A, temp_matrix);
	//
	puts("Start conjugate gradient on GPU...");

	cudaEventRecord(start, 0);
	cublasCreate(&handle);
	cublasSetMathMode(handle, mode);
	cublasSetPointerMode(handle, mode_pt);

	cublasSetMatrix(dimension, dimension, sizeof(cuDoubleComplex), A, dimension, A_dev, dimension);
	cublasSetVector(dimension, sizeof(cuDoubleComplex), b, 1, b_dev, 1);
	cublasSetVector(dimension, sizeof(cuDoubleComplex), x_k_host, 1, x_k_dev, 1);
	alpha = make_cuDoubleComplex(1.0, 0.0);
	beta = make_cuDoubleComplex(0.0, 0.0);

	//convert |b> to A^H|b>
	cublasZgemv(handle, CUBLAS_OP_C, dimension, dimension, &alpha, A_dev, dimension, b_dev, 1, &beta, b_dev, 1);
	//conver A to A^HA
//	cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, dimension, dimension, dimension, &alpha, A_dev, dimension, A_dev, dimension, &beta, A_dev, dimension);
	cublasSetMatrix(dimension, dimension, sizeof(cuDoubleComplex), temp_matrix, dimension, A_dev, dimension);

	cublasDznrm2(handle, dimension, b_dev, 1, &norm);
	cudaMemcpy(r_k_dev, b_dev, dimension*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_k_dev, r_k_dev, dimension*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	error = 1.0;

	while (error>criteria)
	{
		// calculate A|p_k>
		cublasZgemv(handle, CUBLAS_OP_N, dimension, dimension, &alpha, A_dev, dimension, p_k_dev, 1, &beta, b_dev, 1);	//use b_dev to save A|p_k>
		// calculate lambda_k
		cublasDznrm2(handle, dimension, r_k_dev, 1, &error);	
		cublasZdotc(handle, dimension, p_k_dev, 1, b_dev, 1, &lambda_k);	// use error to save norm(A|p_k>)
		lambda_k.x = pow(error,2.)/lambda_k.x;
		lambda_k.y = 0.0;
		// calculate |x_(k+1)>
		cublasZaxpy(handle, dimension, &lambda_k, p_k_dev, 1, x_k_dev, 1);
		// calcualte |r_(k+1)>
		lambda_k.x *= -1.0;
		cublasZaxpy(handle, dimension, &lambda_k, b_dev, 1, r_k_dev, 1);
		// calculate u_k
		cublasZdotc(handle, dimension, r_k_dev, 1, r_k_dev, 1, &u_k);
		u_k.x = u_k.x/pow(error,2.);
		u_k.y = 0.0;
		// calcualte |p_(k+1)>
		cublasZscal(handle, dimension, &u_k, p_k_dev, 1.);
		cublasZaxpy(handle, dimension, &alpha, r_k_dev, 1, p_k_dev, 1);
		// calculate error
		cublasDznrm2(handle, dimension, r_k_dev, 1, &error);
		error /= norm;

		iter += 1;
//		printf("Iteation = %d ; error = %.16e .\n", iter, error);
	}
	printf("Iteation = %d ; error = %.16e .\n", iter, error);

	// get answer
	cublasGetVector(dimension, sizeof(cuDoubleComplex), x_k_dev, 1, x_k_host, 1);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_GPU, start, stop);
	printf("\nGPU total computation time is %.2f ms.\n", time_GPU);

	scanf("%c", &print_out);
	puts("Print solution or not? (y/n)");
	scanf("%c", &print_out);
	if (print_out=='y')
	{
		printf("\nThe GPU solution vector is:\n");
		for (int i=0; i<dimension; i++)
			printf("%.16f%+.16fI\n", x_k_host[i].x, x_k_host[i].y);
		printf("\n");
	}
	else if (print_out!='n')
	{
		puts("Wrong input! Exit!");
		return EXIT_FAILURE;
	}
	// chcek answer
	puts("Do error check for GPU...");
	beta = make_cuDoubleComplex(-1.0, 0.0);
	cublasSetVector(dimension, sizeof(cuDoubleComplex), b, 1, b_dev, 1);
	cublasDznrm2(handle, dimension, b_dev, 1, &norm);
	cublasSetMatrix(dimension, dimension, sizeof(cuDoubleComplex), A, dimension, A_dev, dimension);
	cublasZgemv(handle, CUBLAS_OP_N, dimension, dimension, &alpha, A_dev, dimension, x_k_dev, 1, &beta, b_dev, 1);
	error = 0.0;
	cublasDznrm2(handle, dimension, b_dev, 1, &error);
	cudaMemcpy(r_k_dev, b_dev, dimension*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	printf("The error for GPU is %.16e .\n", error/norm);

	cublasDestroy(handle);
	cudaFree(A_dev);
	cudaFree(b_dev);
	cudaFree(x_k_dev);
	cudaFree(r_k_dev);
	cudaFree(p_k_dev);
	
#if (DO_CPU_CHECK==true)
	puts("\nStart CPU comparsion...");
	cuDoubleComplex temp_scalar;
	cuDoubleComplex *r_k_host, *p_k_host, *temp_vector;
	cudaEventRecord(start,0);

	x_k_host = (cuDoubleComplex *)calloc(dimension, sizeof(cuDoubleComplex));
	r_k_host = (cuDoubleComplex *)malloc(dimension*sizeof(cuDoubleComplex));
	p_k_host = (cuDoubleComplex *)malloc(dimension*sizeof(cuDoubleComplex));
	temp_vector = (cuDoubleComplex *)malloc(dimension*sizeof(cuDoubleComplex));

	//convert |b> to A^H|b>
	memcpy(temp_vector, b, dimension*sizeof(cuDoubleComplex));
	MV_PRODUCT('C', dimension, A, b, b);
	//conver A to A^HA

	memcpy(r_k_host, b, dimension*sizeof(cuDoubleComplex));
	memcpy(p_k_host, r_k_host, dimension*sizeof(cuDoubleComplex));
	norm = sqrt((VV_PRODUCT(dimension, b, b)).x);
	iter = 0;
	error = 1.0;
		
	while (error>criteria)
	{
		MV_PRODUCT('N', dimension, temp_matrix, p_k_host, b);
		temp_scalar = VV_PRODUCT(dimension, p_k_host, b);
		error = (VV_PRODUCT(dimension, r_k_host, r_k_host)).x;
		lambda_k.x = error/temp_scalar.x;
		lambda_k.y = 0.0;
		VV_ADDITION(dimension, &alpha, &lambda_k, x_k_host, p_k_host, x_k_host);
		lambda_k.x *= -1.0;
		VV_ADDITION(dimension, &alpha, &lambda_k, r_k_host, b, r_k_host);
			
		temp_scalar = VV_PRODUCT(dimension, r_k_host, r_k_host);
		u_k.x = temp_scalar.x/error;
		u_k.y = 0.0;
		VV_ADDITION(dimension, &alpha, &u_k, r_k_host, p_k_host, p_k_host);
		
		error = sqrt((VV_PRODUCT(dimension, r_k_host, r_k_host)).x)/norm;
		iter += 1;
		
//		printf("Iteation = %d ; error = %.16e .\n", iter, error);
	}
	printf("Iteation = %d ; error = %.16e .\n", iter, error);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_CPU, start, stop);
	printf("CPU total computation time is %.2f ms.\n", time_CPU);
	printf("The speed up is %.4f .\n", time_CPU/time_GPU);

	if (print_out=='y')
	{
		printf("\nThe CPU solution vector is:\n");
		for (int i=0; i<dimension; i++)
			printf("%.16f%+.16fI\n", x_k_host[i].x, x_k_host[i].y);
		printf("\n");
	}
	else if (print_out!='n')
	{
		puts("Wrong input! Exit!");
		return EXIT_FAILURE;
	}

	puts("Do error check for CPU...");
	beta = make_cuDoubleComplex(-1.0, 0.0);
	norm = sqrt((VV_PRODUCT(dimension, temp_vector, temp_vector)).x);
	MV_PRODUCT('N', dimension, A, x_k_host, b);	
	VV_ADDITION(dimension, &alpha, &beta, b, temp_vector, temp_vector);	// original |b> is saved as |temp_vector>
	error = sqrt((VV_PRODUCT(dimension, temp_vector, temp_vector)).x);
	printf("The error for CPU is %.16e .\n", error/norm);
	
	free(r_k_host);
	free(p_k_host);
	free(temp_vector);
#endif

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(temp_matrix);
	free(x_k_host);
	free(A);
	free(b);
	return EXIT_SUCCESS;
}

void MHM_PRODUCT(int N, cuDoubleComplex *A, cuDoubleComplex *B, cuDoubleComplex *ANS)
{
	int index, index_dummy_A, index_dummy_B;
	cuDoubleComplex *temp = (cuDoubleComplex *)calloc(N*N, sizeof(cuDoubleComplex));
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			index = IDX2C(j,i,N);
			for (int k=0; k<N; k++)
			{
				index_dummy_A = IDX2C(k,j,dimension);
				index_dummy_B = IDX2C(k,i,dimension);
				temp[index].x += A[index_dummy_A].x*B[index_dummy_B].x + A[index_dummy_A].y*B[index_dummy_B].y;
				temp[index].y += A[index_dummy_A].x*B[index_dummy_B].y - A[index_dummy_A].y*B[index_dummy_B].x;
			}
		}
	}
	memcpy(ANS, temp, N*N*sizeof(cuDoubleComplex));
	free(temp);
}

#if (DO_CPU_CHECK==true)
void VV_ADDITION(int N, cuDoubleComplex *alpha, cuDoubleComplex *beta, cuDoubleComplex *a, cuDoubleComplex* b, cuDoubleComplex* ans)
{
	cuDoubleComplex temp;
	for (int i=0; i<N; i++)
	{
		temp.x = ((*alpha).x*a[i].x - (*alpha).y*a[i].y) + ((*beta).x*b[i].x - (*beta).y*b[i].y);
		temp.y = ((*alpha).x*a[i].y + (*alpha).y*a[i].x) + ((*beta).x*b[i].y + (*beta).y*b[i].x);

		ans[i].x = temp.x;
		ans[i].y = temp.y;
	}
}

cuDoubleComplex VV_PRODUCT(int N, cuDoubleComplex *a, cuDoubleComplex *b)
{
	cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
	for (int i=0; i<N; i++)
	{
		sum.x += a[i].x*b[i].x + a[i].y*b[i].y;
		sum.y += a[i].x*b[i].y - a[i].y*b[i].x;
	}
	return(sum);
}

void MV_PRODUCT(char operation, int N, cuDoubleComplex *A, cuDoubleComplex *b, cuDoubleComplex *ans)
{
	int index;
	cuDoubleComplex *temp = (cuDoubleComplex *)calloc(N, sizeof(cuDoubleComplex));
	if (operation=='N')
	{
		for (int i=0; i<N; i++)
		{
			for (int j=0; j<N; j++)
			{
				index = IDX2C(i,j,dimension);
				temp[i].x += A[index].x*b[j].x - A[index].y*b[j].y;
				temp[i].y += A[index].x*b[j].y + A[index].y*b[j].x;
			}
		}
	}
	else if (operation=='C')
	{
		for (int i=0; i<N; i++)
		{
			for (int j=0; j<N; j++)
			{
				index = IDX2C(j,i,dimension);
				temp[i].x += A[index].x*b[j].x + A[index].y*b[j].y;
				temp[i].y += A[index].x*b[j].y - A[index].y*b[j].x;
			}
		}
	}
	else 
	{
		puts("Wrong input! Exit!");
		exit(1);
	}
	memcpy(ans, temp, N*sizeof(cuDoubleComplex));
	free(temp);
}
#endif
