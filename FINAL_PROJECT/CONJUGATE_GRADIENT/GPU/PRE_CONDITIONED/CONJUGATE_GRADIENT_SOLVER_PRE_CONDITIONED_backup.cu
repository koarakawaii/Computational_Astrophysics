#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void FPRINTF(FILE*, int N, double, double*);
double EVALUATE_ERROR(int, int, double*);

__global__ void INITIALIZE(int N, double dx, double* rho_even, double *rho_odd, double* field_even, double* field_odd, double* field_analytic, double* rho_analytic)
{
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N;
	int idx_eo = idx_x/2 + idx_y*(N/2);

	double x = idx_x*dx;
	double y = idx_y*dx;

	field_analytic[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
		
	if (((idx_x%2)+(idx_y%2))%2==0)
	{
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			field_even[idx_eo] = 0.0;
			rho_even[idx_eo] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		}
		else
		{
			field_even[idx_eo] = field_analytic[idx];
			rho_even[idx_eo] = 0.0;
		}
		rho_analytic[idx] = rho_even[idx_eo];	// Notice that rho has been times by dx^2!!
	}
	else
	{
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			field_odd[idx_eo] = 0.0;
			rho_odd[idx_eo] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		}
		else
		{
			field_odd[idx_eo] = field_analytic[idx];
			rho_odd[idx_eo] = 0.0;
		}
		rho_analytic[idx] = rho_odd[idx_eo];	// Notice that rho has been times by dx^2!!
	}
}

__global__ void EVALUATE_ERROR_BLOCK_EVEN(int N, double* rho, double* field_self, double* field_neighbor, double* error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + N/2*idx_y;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;

	if (idx_y!=0&&idx_y!=N-1)
	{
		if (idx_y%2==0)
		{
			if (idx_x!=0)
			{
				int L = idx - 1;
				int R = idx;
				int U = idx + N/2;
				int D = idx - N/2;
				sm[idx_sm] = pow((field_neighbor[L]+field_neighbor[R]+field_neighbor[U]+field_neighbor[D]-4.*field_self[idx])-rho[idx], 2.);
			}
			else
				sm[idx_sm] = 0.0;
		}
		else
		{
			if (idx_x!=N/2-1)
			{
				int L = idx;
				int R = idx + 1;
				int U = idx + N/2;
				int D = idx - N/2;
				sm[idx_sm] = pow((field_neighbor[L]+field_neighbor[R]+field_neighbor[U]+field_neighbor[D]-4.*field_self[idx])-rho[idx], 2.);
			}
			else
				sm[idx_sm] = 0.0;
		}
	}
	else
		sm[idx_sm] = 0.0;
	__syncthreads();

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

__global__ void EVALUATE_ERROR_BLOCK_ODD(int N, double* rho, double* field_self, double* field_neighbor, double* error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + N/2*idx_y;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;

	if (idx_y!=0&&idx_y!=N-1)
	{
		if (idx_y%2==0)
		{
			if (idx_x!=N/2-1)
			{
				int L = idx;
				int R = idx + 1;
				int U = idx + N/2;
				int D = idx - N/2;
				sm[idx_sm] = pow((field_neighbor[L]+field_neighbor[R]+field_neighbor[U]+field_neighbor[D]-4.*field_self[idx])-rho[idx], 2.);
			}
			else
				sm[idx_sm] = 0.0;
		}
		else
		{
			if (idx_x!=0)
			{
				int L = idx - 1;
				int R = idx;
				int U = idx + N/2;
				int D = idx - N/2;
				sm[idx_sm] = pow((field_neighbor[L]+field_neighbor[R]+field_neighbor[U]+field_neighbor[D]-4.*field_self[idx])-rho[idx], 2.);
			}
			else
				sm[idx_sm] = 0.0;
		}
	}
	else
		sm[idx_sm] = 0.0;
	__syncthreads();

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

__global__ void LAPLACIAN_EVEN(int N, double dx, double photon_mass, double* p_self, double* p_neighbor, double* A_p)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	int idx = idx_x + N/2*idx_y;

	if (idx_y!=0&&idx_y!=N-1)
	{
		if (idx_y%2==0)
		{
			if (idx_x!=0)
			{
				int L = idx - 1;
				int R = idx;
				int U = idx + N/2;
				int D = idx - N/2;
				A_p[idx] = (p_neighbor[L]+p_neighbor[R]+p_neighbor[U]+p_neighbor[D]-(4.-pow(photon_mass*dx,2.))*p_self[idx]);
			}
			else
				A_p[idx] = 0.0;
		}
		else
		{
			if (idx_x!=N/2-1)
			{
				int L = idx;
				int R = idx + 1;
				int U = idx + N/2;
				int D = idx - N/2;
				A_p[idx] = (p_neighbor[L]+p_neighbor[R]+p_neighbor[U]+p_neighbor[D]-(4.-pow(photon_mass*dx,2.))*p_self[idx]);
			}
			else
				A_p[idx] = 0.0;
		}
	}
	else
		A_p[idx] = 0.0;
}

__global__ void LAPLACIAN_ODD(int N, double dx, double photon_mass, double* p_self, double* p_neighbor, double* A_p)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	int idx = idx_x + N/2*idx_y;

	if (idx_y!=0&&idx_y!=N-1)
	{
		if (idx_y%2==0)
		{
			if (idx_x!=N/2-1)
			{
				int L = idx;
				int R = idx + 1;
				int U = idx + N/2;
				int D = idx - N/2;
				A_p[idx] = (p_neighbor[L]+p_neighbor[R]+p_neighbor[U]+p_neighbor[D]-(4.-pow(photon_mass*dx,2.))*p_self[idx]);
			}
			else
				A_p[idx] = 0.0;
		}
		else
		{
			if (idx_x!=0)
			{
				int L = idx - 1;
				int R = idx;
				int U = idx + N/2;
				int D = idx - N/2;
				A_p[idx] = (p_neighbor[L]+p_neighbor[R]+p_neighbor[U]+p_neighbor[D]-(4.-pow(photon_mass*dx,2.))*p_self[idx]);
			}
			else
				A_p[idx] = 0.0;
		}
	}
	else
		A_p[idx] = 0.0;
}

__global__ void DAXPY(int N, double c, double* A, double* B)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	int idx = idx_x + N*idx_y;

	A[idx] = c*A[idx] + B[idx];
}

int main(void)
{
	int N, N_block, display_interval, tpb_x, tpb_y, bpg_x, bpg_y, shift, shift_block;
	float preparation_time, computation_time, total_time;
	double photon_mass, omega, dx, criteria;
	double alpha, beta;
	long iter, iter_max;
	double *field, *rho, *r, *p, *A_p, *field_analytic, *rho_analytic, *error_block;
	size_t size_lattice, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;
	printf("Solve the Poission problem using CG by GPU with pre-conditioned.\n\n");
	printf("Enter the latttice size (N,N) (N must be divisible by 2).");
	scanf("%d", &N);
	if (N%2!=0)
	{
		printf("N is not divisible by 2! Exit!\n");
		exit(1);
	}
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("Set the photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("Set the value of omega.\n");
	scanf("%lf", &omega);
	printf("The value of omega is %.4e .\n", omega);
	printf("Set the maximum iteration times.\n");
	scanf("%ld", &iter_max);
	printf("The maximum iteration times is %ld .\n", iter_max);
	printf("Set the stopping criteria.\n");
	scanf("%lf", &criteria);
	printf("The stopping criteria is %.4e .\n", criteria);
	printf("Set the display interval during iterations.\n");
	scanf("%d", &display_interval);
	printf("The display interval is set to be %d .\n", display_interval);
	printf("Set the GPU threads per block (tx,ty). (N/2 must be divisible by tx and N must be divisible by ty)\n");
	scanf("%d %d", &tpb_x, &tpb_y);
	if ((N/2)%tpb_x!=0)
	{
		printf("N/2 is not divisible by tx! Exit!\n");
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
		bpg_x = (N/2)/tpb_x;
		bpg_y = N/tpb_y;
		printf("Blocks per grid for GPU is (%d,%d) .\n", bpg_x, bpg_y);
	}
	printf("\n");

	printf("Start Preparation...\n");
	N_block = 2*bpg_x*bpg_y;
	shift = (N/2)*N;
	shift_block = bpg_x*bpg_y;
	dx = 1./(N-1);	
	size_lattice = N*N*sizeof(double);
	size_sm = tpb_x*tpb_y*sizeof(double);
	output_field = fopen("analytical_field_distribution_CG_pre.txt","w");
	output_rho = fopen("charge_distribution_CG_pre.txt","w");

	cudaSetDevice(0);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 tpb(tpb_x,tpb_y);
	dim3 bpg(bpg_x,bpg_y);
	dim3 bpg_init(2*bpg_x,bpg_y);
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
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));
	cudaMallocManaged(&field_analytic, 2*size_lattice);
	cudaMallocManaged(&rho_analytic, 2*size_lattice);

	INITIALIZE<<<bpg_init,tpb>>>(N, dx, rho, rho+shift, field, field+shift, field_analytic, rho_analytic);
	EVALUATE_ERROR_BLOCK_EVEN<<<bpg,tpb,size_sm>>>(N, rho, field, field+shift, error_block);
	EVALUATE_ERROR_BLOCK_ODD<<<bpg,tpb,size_sm>>>(N, rho+shift, field+shift, field, error_block+shift_block);
	double norm;
	cublasDdot(handle, N*N, rho, 1, rho, 1, &norm);
	norm = sqrt(norm);
	
	cudaDeviceSynchronize();
	cudaMemcpy(r, rho, size_lattice, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p, rho, size_lattice, cudaMemcpyDeviceToDevice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho_analytic);
	cudaEventRecord(start,0);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	double error = EVALUATE_ERROR(N, N_block, error_block); 
	double temp;

	printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
	iter = 0;
	
	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		LAPLACIAN_EVEN<<<bpg,tpb>>>(N, dx, photon_mass, p, p+shift, A_p);
		LAPLACIAN_ODD<<<bpg,tpb>>>(N, dx, photon_mass, p+shift, p, A_p+shift);
		cublasDdot(handle, N*N, p, 1, A_p, 1, &temp);
		alpha = error/temp;
		temp = -alpha;
		cublasDaxpy(handle, N*N, &temp, A_p, 1, r, 1);
		cublasDaxpy(handle, N*N, &alpha, p, 1, field, 1);
		cublasDdot(handle, N*N, r, 1, r, 1, &temp);
		beta = temp/error;
//		printf("%.4f\t%.4f\n", alpha, beta);
		DAXPY<<<bpg_init,tpb>>>(N, beta, p, r);
		error = temp;
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
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
	cudaFree(A_p);
	cudaFree(field_analytic);
	cudaFree(rho_analytic);
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
