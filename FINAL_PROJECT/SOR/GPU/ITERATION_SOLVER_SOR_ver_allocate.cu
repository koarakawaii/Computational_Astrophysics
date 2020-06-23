/* This version use neighbor_e and neighbor_o to stroe the negibhor indices of even sites and odd site.*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void FPRINTF(FILE*, int N, double*);
double EVALUATE_ERROR(int, int, double, double*);

__global__ void INITIALIZE(int N, double dx, int* L_e, int* L_o, int* R_e, int* R_o, int* D_e, int* D_o, int* U_e, int* U_o, double* rho_even, double* rho_odd, double* rho, double* field_even, double* field_odd, double* field_analytic)
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
	
	field_analytic[idx_site_e] = x_e*(1.-x_e)*y_e*(1.-y_e)*exp(x_e-y_e);

	if ( site_x_e==0 || site_x_e==N-1 || site_y_e==0 || site_y_e==N-1 )
	{
		field_even[idx] = field_analytic[idx_site_e];
		rho[idx_site_e] = 0.0;
		rho_even[idx] = 0.0;
	}
	else
	{
		field_even[idx] = 0.0;
		rho[idx_site_e] = 2.*x_e*(y_e-1)*(y_e-2.*x_e+x_e*y_e+2)*exp(x_e-y_e);
		rho_even[idx] = rho[idx_site_e];
//		field_even[idx] = field_analytic[idx_site_o];
	}
	
	field_analytic[idx_site_o] = x_o*(1.-x_o)*y_o*(1.-y_o)*exp(x_o-y_o);

	if ( site_x_o==0 || site_x_o==N-1 || site_y_o==0 || site_y_o==N-1 )
	{
		field_odd[idx] = field_analytic[idx_site_o];
		rho[idx_site_o] = 0.0;
		rho_odd[idx] = 0.0;
	}
	else
	{
		field_odd[idx] = 0.0;
		rho[idx_site_o]= 2.*x_o*(y_o-1)*(y_o-2.*x_o+x_o*y_o+2)*exp(x_o-y_o);
		rho_odd[idx] = rho[idx_site_o];
//		field_odd[idx] = field_analytic[idx_site_o];
	}

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
	
				L_e[idx] = L;
				R_e[idx] = R;
				D_e[idx] = D;
				U_e[idx] = U;
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
			if (site_x!=(N/2)-1)
			{
				int L = idx;	
				int R = site_x +1 + site_y*(N/2);
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				
				L_o[idx] = L;
				R_o[idx] = R;
				D_o[idx] = D;
				U_o[idx] = U;
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
				int U = site_x + (site_y_e+1)*(N/2);
				
				L_e[idx] = L;
				R_e[idx] = R;
				D_e[idx] = D;
				U_e[idx] = U;
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
			if (site_x!=0)
			{
				int L = site_x-1 + site_y*(N/2);	
				int R = idx;
				int D = site_x + (site_y-1)*(N/2);
				int U = site_x + (site_y+1)*(N/2);
				
				L_o[idx] = L;
				R_o[idx] = R;
				D_o[idx] = D;
				U_o[idx] = U;
//				printf("%d\t%d\t%d\t%d\t%d\n", idx, L, R, U, D);
			}
		}
	}
	else
	{
		L_e[idx] = 0;
		R_e[idx] = 0;
		U_e[idx] = 0;
		D_e[idx] = 0;
		L_o[idx] = 0;
		R_o[idx] = 0;
		U_o[idx] = 0;
		D_o[idx] = 0;
	}
}

__global__ void SOR_SOLVER_EVEN(int N, double dx, double omega, int* L_e, int* R_e, int* D_e, int* U_e, double* field_even, double* field_odd, double* rho_even)
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
				field_even[idx] += 0.25*omega*( field_odd[L_e[idx]] + field_odd[R_e[idx]] + field_odd[U_e[idx]] + field_odd[D_e[idx]] - dx*dx*rho_even[idx] - 4.*field_even[idx]);
			}
		}		
		else
		{
			if (site_x!=(N/2)-1)
			{
				field_even[idx] += 0.25*omega*( field_odd[L_e[idx]] + field_odd[R_e[idx]] + field_odd[U_e[idx]] + field_odd[D_e[idx]] - dx*dx*rho_even[idx] - 4.*field_even[idx]);
			}
		}		
	}
	__syncthreads();
}

__global__ void SOR_SOLVER_ODD(int N, double dx, double omega, int* L_o, int* R_o, int* D_o, int* U_o, double* field_even, double* field_odd, double* rho_odd)
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
				field_odd[idx] += 0.25*omega*( field_even[L_o[idx]] + field_even[R_o[idx]] + field_even[U_o[idx]] + field_even[D_o[idx]] - dx*dx*rho_odd[idx] - 4.*field_odd[idx]);
			}
		}		
		else
		{
			if (site_x!=0)
			{
				field_odd[idx] += 0.25*omega*( field_even[L_o[idx]] + field_even[R_o[idx]] + field_even[U_o[idx]] + field_even[D_o[idx]] - dx*dx*rho_odd[idx] - 4.*field_odd[idx]);
			}
		}		
	}
	__syncthreads();
}

__global__ void ERROR(int N, double dx, int* L_e, int* L_o, int* R_e, int* R_o, int* D_e, int* D_o, int* U_e, int* U_o, double* rho_even, double* rho_odd, double* field_even, double* field_odd, double *error_block)
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
				sm[idx_sm] += pow((field_odd[L_e[idx]]+field_odd[R_e[idx]]+field_odd[D_e[idx]]+field_odd[U_e[idx]]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
			if (site_x!=(N/2)-1)
				sm[idx_sm] += pow((field_even[L_o[idx]]+field_even[R_o[idx]]+field_even[D_o[idx]]+field_even[U_o[idx]]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);
			
		}		
		else
		{
			if (site_x!=(N/2)-1)
				sm[idx_sm] += pow((field_odd[L_e[idx]]+field_odd[R_e[idx]]+field_odd[D_e[idx]]+field_odd[U_e[idx]]-4.0*field_even[idx])/dx/dx-rho_even[idx], 2.0);
			if (site_x!=0)
				sm[idx_sm] += pow((field_even[L_o[idx]]+field_even[R_o[idx]]+field_even[D_o[idx]]+field_even[U_o[idx]]-4.0*field_odd[idx])/dx/dx-rho_odd[idx], 2.0);

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
	int **neighbor_even, **neighbor_odd;
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
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	dx = 1./(N-1);	
	N_block = (N/2)/tpb_x*(N/tpb_y);
	size_lattice = N*N*sizeof(double);
	size_sm = tpb_x*tpb_y*sizeof(double);
	field_final = (double*)malloc(N*N*sizeof(double));
	neighbor_even = (int**)malloc(4*sizeof(int*));
	neighbor_odd = (int**)malloc(4*sizeof(int*));
	output_field = fopen("analytical_field_distribution_allocate.txt","w");
	output_rho = fopen("charge_distribution_allocate.txt","w");

	cudaSetDevice(0);
	dim3 tpb(tpb_x,tpb_y);
	dim3 bpg(bpg_x,bpg_y);

	cublasMath_t mode = CUBLAS_TENSOR_OP_MATH;
    cublasPointerMode_t mode_pt = CUBLAS_POINTER_MODE_HOST;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, mode);
    cublasSetPointerMode(handle, mode_pt);

	cudaMallocManaged(&field_even, size_lattice/2);
	cudaMallocManaged(&field_odd, size_lattice/2);
	cudaMallocManaged(&field_analytic, size_lattice);
	cudaMallocManaged(&rho_even, size_lattice/2);
	cudaMallocManaged(&rho_odd, size_lattice/2);
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));

	// construct neighbor index
	for (int i=0; i<4; i++)
	{
		cudaMallocManaged(&neighbor_even[i], (N*N)/2*sizeof(int));
		cudaMallocManaged(&neighbor_odd[i], (N*N)/2*sizeof(int));
	}
	//
	
	INITIALIZE<<<bpg,tpb,size_sm>>>(N, dx, neighbor_even[0], neighbor_odd[0], neighbor_even[1], neighbor_odd[1], neighbor_even[2],neighbor_odd[2], neighbor_even[3], neighbor_odd[3], rho_even, rho_odd, rho, field_even, field_odd, field_analytic);
	ERROR<<<bpg,tpb,size_sm>>>(N, dx, neighbor_even[0], neighbor_odd[0], neighbor_even[1], neighbor_odd[1], neighbor_even[2], neighbor_odd[2], neighbor_even[3], neighbor_odd[3], rho_even, rho_odd, field_even, field_odd, error_block);
	cudaDeviceSynchronize();

	double norm;
	cublasDdot(handle, N*N, rho, 1, rho, 1, &norm);
    norm = sqrt(norm);
	printf("Norm = %.4e\n", norm);

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
	double error = EVALUATE_ERROR(N, N_block, norm, error_block); 

	printf("Starts computation with error = %.8e...\n", error);
	iter = 0;
	while (error>criteria&&iter<iter_max)
	{
		SOR_SOLVER_EVEN<<<bpg,tpb>>>(N, dx, omega, neighbor_even[0], neighbor_even[1], neighbor_even[2], neighbor_even[3], field_even, field_odd, rho_even);
//		cudaDeviceSynchronize();
		SOR_SOLVER_ODD<<<bpg,tpb>>>(N, dx, omega, neighbor_odd[0], neighbor_odd[1], neighbor_odd[2], neighbor_odd[3], field_even, field_odd, rho_odd);
//		cudaDeviceSynchronize();
		ERROR<<<bpg,tpb,size_sm>>>(N, dx, neighbor_even[0], neighbor_odd[0], neighbor_even[1], neighbor_odd[1], neighbor_even[2], neighbor_odd[2], neighbor_even[3], neighbor_odd[3], rho_even, rho_odd, field_even, field_odd, error_block);
		cudaDeviceSynchronize();
		error = EVALUATE_ERROR(N, N_block, norm, error_block);
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
  
	output_field = fopen("simulated_field_distribution_GPU_allocate.txt","w");
	FPRINTF(output_field, N, field_final);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

	free(field_final);
	free(neighbor_even);
	free(neighbor_odd);
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

double EVALUATE_ERROR(int N, int N_block, double norm, double* error_block)
{
	double error = 0.0;
	for (int i=0; i<N_block; i++)
		error += error_block[i];
	return sqrt(error)/norm;
}

//void LAPLACIAN_SOR(int N, double dx, double omega, double* field_even, double* field_odd, double *rho_even, double *rho_odd, int **neighbor_even, int **neighbor_odd)
//{
//#	pragma omp parallel for
//	for (int i_E=N/2; i_E<(N*N)/2-N/2; i_E++)
//	{
//		int i_x = i_E%(N/2);
//		int i_y = i_E/(N/2);
//		if (i_y%2==0)
//		{
//			if (i_x!=0)
//				field_even[i_E] += 0.25*omega*( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
//		}		
//		else
//		{
//			if (i_x!=(N/2)-1)
//				field_even[i_E] += 0.25*omega*( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
//		}
//	}
//#	pragma omp parallel for
//	for (int i_O=N/2; i_O<(N*N)/2-N/2; i_O++)
//	{
//		int i_x = i_O%(N/2);
//		int i_y = i_O/(N/2);
//		if (i_y%2==0)
//		{
//			if (i_x!=(N/2)-1)
//				field_odd[i_O] += 0.25*omega*( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
//		}		
//		else
//		{
//			if (i_x!=0)
//				field_odd[i_O] += 0.25*omega*( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
//		}
//	}
//}

void FPRINTF(FILE *output_file, int N, double *array)
{
	for (int j=0; j<N; j++)
	{
		for (int i=0; i<N; i++)
			fprintf(output_file, "%.8e\t", array[i+j*N]);
		fprintf(output_file, "\n");
	}

}
