/* This version calculates the negibhor indices of even sites and odd site.*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#define neighbor_dim 2048000

void FPRINTF(FILE*, int N, double*);
double EVALUATE_ERROR(int, int, double*);

__global__ void INITIALIZE(int N, double dx, double* rho_even, double* rho_odd, double* rho, double* field_even, double* field_odd, double* field_analytic, double *error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N/2;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;
	int idx_site_e, idx_site_o;
	sm[idx_sm] = 0.0;

	int site_x_e = (2*idx)%N;		
	int site_y_e = (2*idx)/N;
	int parity_e = (site_x_e+site_y_e)%2;
	site_x_e += parity_e;
	double x_e = site_x_e*dx;
	double y_e = site_y_e*dx;
	idx_site_e = site_x_e + N*site_y_e;

	int site_x_o = (2*idx)%N;		
	int site_y_o = (2*idx)/N;
	int parity_o = (site_x_o+site_y_o+1)%2;
	site_x_o += parity_o;
	double x_o = site_x_o*dx;
	double y_o = site_y_o*dx;
	idx_site_o = site_x_o + N*site_y_o;
	
	rho[idx_site_e]= 2.*x_e*(y_e-1)*(y_e-2.*x_e+x_e*y_e+2)*exp(x_e-y_e);
	rho_even[idx] = rho[idx_site_e];
	field_analytic[idx_site_e] = x_e*(1.-x_e)*y_e*(1.-y_e)*exp(x_e-y_e);

	if ( site_x_e==0 || site_x_e==N-1 || site_y_e==0 || site_y_e==N-1 )
		field_even[idx] = field_analytic[idx_site_e];
	else
		field_even[idx] = 0.0;
//		field_even[idx] = field_analytic[idx_site_o];
	
	rho[idx_site_o]= 2.*x_o*(y_o-1)*(y_o-2.*x_o+x_o*y_o+2)*exp(x_o-y_o);
	rho_odd[idx] = rho[idx_site_o];
	field_analytic[idx_site_o] = x_o*(1.-x_o)*y_o*(1.-y_o)*exp(x_o-y_o);

	if ( site_x_o==0 || site_x_o==N-1 || site_y_o==0 || site_y_o==N-1 )
		field_odd[idx] = field_analytic[idx_site_o];
	else
		field_odd[idx] = 0.0;
//		field_odd[idx] = field_analytic[idx_site_o];

	// construct neighbors for even sites
	int site_x = idx%(N/2);
	int site_y = idx/(N/2);
	if ( (idx>N/2-1)&&(idx<(N*N)/2-N/2))
	{
		if (site_y%2==0)
		{
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);
				int R = idx;	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_odd[L]+field_odd[R]+field_odd[D]+field_odd[U]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
			if (site_x!=(N/2)-1)
			{
				int L = idx;	
				int R = site_x +1 + site_y*(N/2);
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_even[L]+field_even[R]+field_even[D]+field_even[U]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
		}		
		else
		{
			if (site_x!=(N/2)-1)
			{
				int L = idx;
				int R = site_x+1 + site_y*(N/2);	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_odd[L]+field_odd[R]+field_odd[D]+field_odd[U]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);	
				int R = idx;
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_even[L]+field_even[R]+field_even[D]+field_even[U]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
		}
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

__global__ void SOR_SOLVER_EVEN(int N, double dx, double omega, double* field_even, double* field_odd, double* rho_even)
{
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N/2;

	int site_x = idx%(N/2);		
	int site_y = idx/(N/2);

	if ( (idx>N/2-1)&&(idx<(N*N)/2-N/2))
	{
		if (site_y%2==0)
		{
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);
				int R = idx;	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);

				field_even[idx] += 0.25*omega*( field_odd[L] + field_odd[R] + field_odd[U] + field_odd[D] - dx*dx*rho_even[idx] - 4.*field_even[idx]);
			}
		}		
		else
		{
			if (site_x!=(N/2)-1)
			{
				int L = idx;
				int R = site_x+1 + site_y*(N/2);	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);

				field_even[idx] += 0.25*omega*( field_odd[L] + field_odd[R] + field_odd[U] + field_odd[D] - dx*dx*rho_even[idx] - 4.*field_even[idx]);
			}
		}		
	}
	__syncthreads();
}

__global__ void SOR_SOLVER_ODD(int N, double dx, double omega, double* field_even, double* field_odd, double* rho_odd)
{
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N/2;

	int site_x = idx%(N/2);		
	int site_y = idx/(N/2);

	if ( (idx>N/2-1)&&(idx<(N*N)/2-N/2))
	{
		if (site_y%2==0)
		{
			if (site_x!=(N/2)-1)
			{
				int L = idx;
				int R = site_x+1 + site_y*(N/2);	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				field_odd[idx] += 0.25*omega*( field_even[L] + field_even[R] + field_even[U] + field_even[D] - dx*dx*rho_odd[idx] - 4.*field_odd[idx]);
			}
		}		
		else
		{
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);	
				int R = idx;
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				field_odd[idx] += 0.25*omega*( field_even[L] + field_even[R] + field_even[U] + field_even[D] - dx*dx*rho_odd[idx] - 4.*field_odd[idx]);
			}
		}		
	}
	__syncthreads();
}

__global__ void ERROR(int N, double dx, double* rho_even, double* rho_odd, double* field_even, double* field_odd, double *error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = idx_x + idx_y*N/2;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;
	sm[idx_sm] = 0.0;

	int site_x = idx%(N/2);
	int site_y = idx/(N/2);
	if ( (idx>N/2-1)&&(idx<(N*N)/2-N/2))
	{
		if (site_y%2==0)
		{
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);
				int R = idx;	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_odd[L]+field_odd[R]+field_odd[D]+field_odd[U]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
			}
			if (site_x!=(N/2)-1)
			{
				int L = idx;
				int R = site_x+1 + site_y*(N/2);	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_even[L]+field_even[R]+field_even[D]+field_even[U]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);
			}
		}		
		else
		{
			if (site_x!=(N/2)-1)
			{
				int L = idx;
				int R = site_x+1 + site_y*(N/2);	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_odd[L]+field_odd[R]+field_odd[D]+field_odd[U]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
			}
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);
				int R = idx;	
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				sm[idx_sm] += pow((field_even[L]+field_even[R]+field_even[D]+field_even[U]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);
			}
		}
	}
	__syncthreads();

	for (int shift=blockDim.x*blockDim.y/2; shift>0; shift/=2)
	{
		if (idx_sm<shift)
			sm[idx_sm] += sm[idx_sm+shift];
		__syncthreads();
	}
	if (idx_sm==0)
		error_block[blockIdx.x+blockIdx.y*gridDim.x] = sm[0];
}

int main(void)
{
	int N, N_threads, N_block, display_interval, tpb_x, tpb_y, bpg_x, bpg_y;
//	int N, N_threads, display_interval, tpb, bpg;
	float preparation_time, computation_time, total_time;
	double omega, dx, criteria;
	long iter, iter_max;
	double *field_even, *field_odd, *rho_even, *rho_odd, *field_final, *field_analytic, *rho, *error_block;
	size_t size_lattice, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;
	printf("Solve the Poission problem using SOR by OpenMP.\n\n");
	printf("Enter the latttice size (N,N) (N must be divisible by 2).");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("Set the value of omega.\n");
	scanf("%lf",&omega);
	printf("The value of omega is %.4f .\n", omega);
	printf("Set the maximum iteration times.\n");
	scanf("%ld", &iter_max);
	printf("The maximum iteration times is %ld .\n", iter_max);
	printf("Set the stopping criteria.\n");
	scanf("%lf", &criteria);
	printf("The stopping criteria is %.4e .\n", criteria);
	printf("Set the display interval during iterations.\n");
	scanf("%d", &display_interval);
	printf("The display interval is set to be %d .\n", display_interval);
//	printf("Set the number of OpenMP threads.\n");
//	scanf("%d", &N_threads);
//	printf("The number of OpenMP threads is %d .\n", N_threads);
	printf("Set the GPU threads per block (tx,ty). (N/2 must be divisible by tx and N must be divisible by N)\n");
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
	printf("Set the number of OpenMP threads.\n");
	scanf("%d", &N_threads);
	printf("The number of OpenMP threads is %d.\n", N_threads);
	printf("\n");

	printf("Start Preparation...\n");
	dx = 1./(N-1);	
	N_block = (N/2)/tpb_x*(N/tpb_y);
	size_lattice = N*N*sizeof(double);
	size_sm = tpb_x*tpb_y*sizeof(double);
	field_final = (double*)malloc(N*N*sizeof(double));
	output_field = fopen("analytical_field_distribution_calculate.txt","w");
	output_rho = fopen("charge_distribution_calculate.txt","w");

	cudaSetDevice(0);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 tpb(tpb_x,tpb_y);
	dim3 bpg(bpg_x,bpg_y);

	cudaEventRecord(start,0);
	cudaMallocManaged(&field_even, size_lattice/2);
	cudaMallocManaged(&field_odd, size_lattice/2);
	cudaMallocManaged(&field_analytic, size_lattice);
	cudaMallocManaged(&rho_even, size_lattice/2);
	cudaMallocManaged(&rho_odd, size_lattice/2);
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));

	INITIALIZE<<<bpg,tpb,size_sm>>>(N, dx, rho_even, rho_odd, rho, field_even, field_odd, field_analytic, error_block);
	cudaDeviceSynchronize();

	// debug
//	for (int j=0; j<N; j++)
//	{
//		for (int i=0; i<N/2; i++)
//			printf("%d\t",neighbor_odd[3][i+N/2*j]);
//		printf("\n");
//	}
	//

	FPRINTF(output_field, N, field_analytic);
	FPRINTF(output_rho, N, rho);
	cudaEventRecord(start,0);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	double error = EVALUATE_ERROR(N, N_block, error_block); 

	printf("Starts computation with error = %.8e...\n", error);
	iter = 0;
	while (error>criteria&&iter<iter_max)
	{
		SOR_SOLVER_EVEN<<<bpg,tpb>>>(N, dx, omega, field_even, field_odd, rho_even);
//		cudaDeviceSynchronize();
		SOR_SOLVER_ODD<<<bpg,tpb>>>(N, dx, omega, field_even, field_odd, rho_odd);
//		cudaDeviceSynchronize();
		ERROR<<<bpg,tpb,size_sm>>>(N, dx, rho_even, rho_odd, field_even, field_odd, error_block);
		cudaDeviceSynchronize();
		error = EVALUATE_ERROR(N, N_block, error_block);
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, error);
	}

	omp_set_num_threads(N_threads);
#	pragma omp parallel for 
		for (int i_E=0; i_E<(N*N)/2; i_E++)
		{
			int ix = (2*i_E)%N;		
			int iy = (2*i_E)/N;
			int parity = (ix+iy)%2;
			ix += parity;
			field_final[ix+iy*N] = field_even[i_E];
		}
#	pragma omp parallel for 
		for (int i_O=0; i_O<(N*N)/2; i_O++)
		{
			int ix = (2*i_O)%N;		
			int iy = (2*i_O)/N;
			int parity = (ix+iy+1)%2;
			ix += parity;
			field_final[ix+iy*N] = field_odd[i_O];
		}
  
	output_field = fopen("simulated_field_distribution_GPU_calculate.txt","w");
	FPRINTF(output_field, N, field_final);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

	free(field_final);
	cudaFree(field_even);
	cudaFree(field_odd);
	cudaFree(field_analytic);
	cudaFree(rho_even);
	cudaFree(rho_odd);
	cudaFree(rho);
	cudaFree(error_block);
	fclose(output_field);
	fclose(output_rho);
	return EXIT_SUCCESS;
}

double EVALUATE_ERROR(int N, int N_block, double* error_block)
{
	double error = 0.0;
	for (int i=0; i<N_block; i++)
		error += error_block[i];
	return sqrt(error/pow(N-2,2.));
}

void FPRINTF(FILE *output_file, int N, double *array)
{
	for (int j=0; j<N; j++)
	{
		for (int i=0; i<N; i++)
			fprintf(output_file, "%.8e\t", array[i+j*N]);
		fprintf(output_file, "\n");
	}

}
