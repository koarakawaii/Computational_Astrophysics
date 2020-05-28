#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void FPRINTF(FILE*, int N, double, double*);
double EVALUATE_ERROR(int, int, double*);

__global__ void INITIALIZE(int N, double dx, double* rho, double* field, double* field_analytic, double* error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;

	double x = idx_x*dx;
	double y = idx_y*dx;

	field_analytic[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
		
	if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
	{
		int L = idx_x-1 + idx_y*N;
		int R = idx_x+1 + idx_y*N;
		int U = idx_x + (idx_y+1)*N;
		int D = idx_x + (idx_y-1)*N;
		field[idx] = 0.0;
		rho[idx] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		sm[idx_sm] = pow((field[L]+field[R]+field[U]+field[D]-4.*field[idx])-rho[idx], 2.);
	}
	else
	{
		field[idx] = field_analytic[idx];
		rho[idx] = 0.0;
		sm[idx_sm] = 0.0;
	}
	__syncthreads();
//	printf("%d\t%d\t%.4f\n", idx, idx_sm, sm[idx_sm]);
	
	for (int shift=blockDim.x*blockDim.y/2; shift>0; shift/=2)
	{
		if (idx_sm<shift)
			sm[idx_sm] += sm[idx_sm+shift];
		__syncthreads();
	}
	if (idx_sm==0)
		error_block[blockIdx.x+blockIdx.y*gridDim.x] = sm[0];
//	printf("%d\t%.4f\n", blockIdx.x+gridDim.x*blockIdx.y, sm[0]);
}

__global__ void LAPLACIAN(int N, double dx, double photon_mass, double* p, double* A_p)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	int idx = idx_x + N*idx_y;
	
	if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
	{
		int L = idx_x-1 + idx_y*N;
		int R = idx_x+1 + idx_y*N;
		int U = idx_x + (idx_y+1)*N;
		int D = idx_x + (idx_y-1)*N;

		A_p[idx] = (p[L]+p[R]+p[U]+p[D]-(4.-pow(photon_mass*dx,2.))*p[idx]);
//		printf("%d\t%.4f\n", idx, A_p[idx]);
	}
	else
		A_p[idx] = 0.0;
}

__global__ void DAXPY(int N, double c, double *A, double *B)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	int idx = idx_x + N*idx_y;

	A[idx] = c*A[idx] + B[idx];
}

int main(void)
{
	int N, N_block, display_interval, tpb_x, tpb_y, bpg_x, bpg_y;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria;
	double alpha, beta;
	long iter, iter_max;
	double *field, *rho, *r, *p, *A_p, *field_analytic, *error_block;
	size_t size_lattice, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;
	printf("Solve the Poission problem using SOR by OpenMP.\n\n");
	printf("Enter the latttice size (N,N) (N must be divisible by 2).");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("Set the photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("Set the maximum iteration times.\n");
	scanf("%ld", &iter_max);
	printf("The maximum iteration times is %ld .\n", iter_max);
	printf("Set the stopping criteria.\n");
	scanf("%lf", &criteria);
	printf("The stopping criteria is %.4e .\n", criteria);
	printf("Set the display interval during iterations.\n");
	scanf("%d", &display_interval);
	printf("The display interval is set to be %d .\n", display_interval);
	printf("Set the GPU threads per block (tx,ty). (N must be divisible by tx and N must be divisible by N)\n");
	scanf("%d %d", &tpb_x, &tpb_y);
	if (N%tpb_x!=0)
	{
		printf("N is not divisible by tx! Exit!\n");
		return EXIT_FAILURE;
	}
	else if (N%tpb_y!=0)
	{
		printf("N is not divisible by ty! Exit!\n");
		return EXIT_FAILURE;
	}
	else
	{
		printf("Threads per block for GPU is (%d,%d) .\n", tpb_x, tpb_y);
		printf("The block per grid will be set automatically.");
		bpg_x = N/tpb_x;
		bpg_y = N/tpb_y;
		printf("Blocks per grid for GPU is (%d,%d) .\n", bpg_x, bpg_y);
	}
	printf("\n");

	printf("Start Preparation...\n");
	dx = 1./(N-1);	
	N_block = (N/tpb_x)*(N/tpb_y);
	size_lattice = N*N*sizeof(double);
	size_sm = tpb_x*tpb_y*sizeof(double);
	output_field = fopen("analytical_field_distribution_CG.txt","w");
	output_rho = fopen("charge_distribution_CG.txt","w");

	cudaSetDevice(0);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 tpb(tpb_x,tpb_y);
	dim3 bpg(bpg_x,bpg_y);
	cublasMath_t mode = CUBLAS_TENSOR_OP_MATH;
    cublasPointerMode_t mode_pt = CUBLAS_POINTER_MODE_HOST;
	cublasHandle_t handle;

	cublasCreate(&handle);
	cublasSetMathMode(handle, mode);
    cublasSetPointerMode(handle, mode_pt);

	cudaEventRecord(start,0);
	cudaMallocManaged(&field, size_lattice);
	cudaMallocManaged(&r, size_lattice);
	cudaMallocManaged(&p, size_lattice);
	cudaMallocManaged(&A_p, size_lattice);
	cudaMallocManaged(&field_analytic, size_lattice);
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));

	INITIALIZE<<<bpg,tpb,size_sm>>>(N, dx, rho, field, field_analytic, error_block);
	cudaDeviceSynchronize();
	cudaMemcpy(r, rho, size_lattice, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p, rho, size_lattice, cudaMemcpyDeviceToDevice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);
	cudaEventRecord(start,0);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	double error = EVALUATE_ERROR(N, N_block, error_block); 
	double temp;

	printf("Starts computation with error = %.8e...\n", error);
	iter = 0;
	
	while (sqrt(error)/(double)(N-2)>criteria&&iter<iter_max)
	{
		LAPLACIAN<<<bpg,tpb>>>(N, dx, photon_mass, p, A_p);
		cublasDdot(handle, N*N, p, 1, A_p, 1, &temp);
		alpha = error/temp;
		temp = -alpha;
		cublasDaxpy(handle, N*N, &temp, A_p, 1, r, 1);
		cublasDaxpy(handle, N*N, &alpha, p, 1, field, 1);
		cublasDdot(handle, N*N, r, 1, r, 1, &temp);
		beta = temp/error;
//		printf("%.4f\t%.4f\n", alpha, beta);
		DAXPY<<<bpg,tpb>>>(N, beta, p, r);
		error = temp;
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/(double)(N-2));
	}
  
	output_field = fopen("simulated_field_distribution_GPU_CG.txt","w");
	FPRINTF(output_field, N, 1., field);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

	cudaFree(field);
	cudaFree(r);
	cudaFree(p);
	cudaFree(field_analytic);
	cudaFree(rho);
	cudaFree(error_block);
	cublasDestroy(handle);
	fclose(output_field);
	fclose(output_rho);
	return EXIT_SUCCESS;
}

double EVALUATE_ERROR(int N, int N_block, double* error_block)
{
	double error = 0.0;
	for (int i=0; i<N_block; i++)
		error += error_block[i];
	return error;
}

void FPRINTF(FILE *output_file, int N, double scale, double *array)
{
	for (int j=0; j<N; j++)
	{
		for (int i=0; i<N; i++)
			fprintf(output_file, "%.8e\t", scale*array[i+j*N]);
		fprintf(output_file, "\n");
	}
}
