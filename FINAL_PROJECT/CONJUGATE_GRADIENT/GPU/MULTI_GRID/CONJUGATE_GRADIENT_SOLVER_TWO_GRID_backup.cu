#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

double EVALUATE_ERROR(int, int, double*);
void FPRINTF(FILE*, int N, double, double*);

__global__ void INITIALIZE(int N, double dx, double* rho, double* field, double* field_analytic)
{
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + idx_y*N;
	
		double x = idx_x*dx;
		double y = idx_y*dx;
	
		field_analytic[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
			
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			field[idx] = 0.0;
			rho[idx] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		}
		else
		{
			field[idx] = field_analytic[idx];
			rho[idx] = 0.0;
		}
	}
}

__global__ void EVALUATE_ERROR_BLOCK(int N, double* rho, double* field, double* error_block)
{
	extern __shared__ double sm[];
	int idx_x = threadIdx.x + blockIdx.x*blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y*blockDim.y;
	int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;
	sm[idx_sm] = 0.0;
	__syncthreads();

	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + N*idx_y;
		int idx_sm = threadIdx.x + blockDim.x*threadIdx.y;
	
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			int L = idx_x-1 + idx_y*N;
			int R = idx_x+1 + idx_y*N;
			int U = idx_x + (idx_y+1)*N;
			int D = idx_x + (idx_y-1)*N;
			sm[idx_sm] = pow((field[L]+field[R]+field[U]+field[D]-4.*field[idx])-rho[idx], 2.);
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
}

__global__ void LAPLACIAN(int N, double dx, double photon_mass, double factor, double* p, double* A_p)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + N*idx_y;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			int L = idx_x-1 + idx_y*N;
			int R = idx_x+1 + idx_y*N;
			int U = idx_x + (idx_y+1)*N;
			int D = idx_x + (idx_y-1)*N;
	
			A_p[idx] = factor*(p[L]+p[R]+p[U]+p[D]-(4.+pow(photon_mass*dx,2.))*p[idx]);
	//		printf("%d\t%.4f\n", idx, A_p[idx]);
		}
		else
			A_p[idx] = factor*p[idx];
	}
}

__global__ void DAXPY(int N, double c, double *A, double *B)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + N*idx_y;
	
		A[idx] = c*A[idx] + B[idx];
	}
}

__global__ void INTERPOLATE_2D(int dimension, double* field_coarse, double* field_fine)
{
	int N_fine = dimension;
	int idx_x_fine = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y_fine = threadIdx.y + blockDim.y*blockIdx.y;
	
	if (idx_x_fine<N_fine&&idx_y_fine<N_fine)
	{
		int idx_fine = idx_x_fine + N_fine*idx_y_fine;
		int N_coarse = (N_fine-1)/2 + 1;
		int idx_x_coarse = idx_x_fine/2;
		int idx_y_coarse = idx_y_fine/2;
		int idx_coarse = idx_x_coarse + N_coarse*idx_y_coarse;

		if (idx_x_fine%2==0&&idx_y_fine%2==0)
			field_fine[idx_fine] = field_coarse[idx_coarse];
		else if (idx_x_fine%2==1&&idx_y_fine%2==0)
			field_fine[idx_fine] = 0.5*(field_coarse[idx_coarse]+field_coarse[idx_coarse+1]);
		else if (idx_x_fine%2==0&&idx_y_fine%2==1)
			field_fine[idx_fine] = 0.5*(field_coarse[idx_coarse]+field_coarse[idx_coarse+N_coarse]);
		else
			field_fine[idx_fine] = 0.25*(field_coarse[idx_coarse]+field_coarse[idx_coarse+1]+field_coarse[idx_coarse+N_coarse]+field_coarse[idx_coarse+N_coarse+1]);
	}
}

__global__ void RESTRICT_2D(int dimension, double* field_fine, double* field_coarse)
{
	int N_coarse = dimension;
	int idx_x_coarse = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y_coarse = threadIdx.y + blockDim.y*blockIdx.y;

	if (idx_x_coarse<N_coarse&&idx_y_coarse<N_coarse)
	{
		int idx_coarse = idx_x_coarse + N_coarse*idx_y_coarse;
		int N_fine = (N_coarse-1)*2 + 1;
		int idx_x_fine = idx_x_coarse*2;
		int idx_y_fine = idx_y_coarse*2;
		int idx_fine = idx_x_fine + idx_y_fine*N_fine;

		if (idx_x_coarse!=0&&idx_x_coarse!=N_coarse-1&&idx_y_coarse!=0&&idx_y_coarse!=N_coarse-1)
			field_coarse[idx_coarse] = 1./16.*(field_fine[idx_fine-4]+field_fine[idx_fine-2]+field_fine[idx_fine+2]+field_fine[idx_fine+4]) + 1./8.*(field_fine[idx_fine-3]+field_fine[idx_fine-1]+field_fine[idx_fine+1]+field_fine[idx_fine+3]) + 1./4.*field_fine[idx_fine];
		else 
			field_coarse[idx_coarse] = field_fine[idx_fine];
	}
}

int main(void)
{
	int N, N_level, N_block, display_interval, tpb_x, tpb_y, bpg_x, bpg_y, row;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria;
	double alpha, beta, error, factor;
	long iter, iter_max;
    int *dimension_level;
	double *rho, *field_analytic, *error_block;
    double **field_level, **r_level, **p_level, **A_p_level;
	size_t size_lattice, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;

	printf("Solve the Poission problem using CG by GPU.\n\n");
	printf("Enter the latttice size (N,N) .");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("The depth of the V process level will be set to 1 (Two-Grids).\n");
    N_level = 1;
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
	printf("Set the GPU threads per block (tx,ty). \n");
	scanf("%d %d", &tpb_x, &tpb_y);
	printf("Threads per block for GPU is (%d,%d) .\n", tpb_x, tpb_y);
	printf("The block per grid will be set automatically.");
	bpg_x = (N+tpb_x-1)/tpb_x;
	bpg_y = (N+tpb_y-1)/tpb_y;
	printf("Blocks per grid for GPU is (%d,%d) .\n", bpg_x, bpg_y);
	printf("\n");

	printf("Start Preparation...\n");
	dx = 1./(N-1);	
	N_block = bpg_x*bpg_y;
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
//	cudaMallocManaged(&field, size_lattice);
//	cudaMallocManaged(&r, size_lattice);
//	cudaMallocManaged(&p, size_lattice);
//	cudaMallocManaged(&A_p, size_lattice);
	cudaMallocManaged(&field_analytic, size_lattice);
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));
	cudaMallocManaged(&dimension_level, (N_level+1)*sizeof(int));

	/* allocate the memory for multi-grid */
    field_level = (double**)malloc((N_level+1)*sizeof(double*));
    r_level = (double**)malloc((N_level+1)*sizeof(double*));
    p_level = (double**)malloc((N_level+1)*sizeof(double*));
    A_p_level = (double**)malloc((N_level+1)*sizeof(double*));
    int dimension = N-1;
    for (int level=0; level<=N_level; level++)
    {
        cudaMallocManaged(&field_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        cudaMallocManaged(&r_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        cudaMallocManaged(&p_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        cudaMallocManaged(&A_p_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        dimension_level[level] = dimension + 1;
        dimension /= 2;
    }

	INITIALIZE<<<bpg,tpb>>>(N, dx, rho, field_level[0], field_analytic);
	EVALUATE_ERROR_BLOCK<<<bpg,tpb,size_sm>>>(N, rho, field_level[0], error_block);
	double norm;
	cublasDdot(handle, N*N, rho, 1, rho, 1, &norm);
	norm = sqrt(norm);
	
	cudaDeviceSynchronize();
	cudaMemcpy(r_level[0], rho, size_lattice, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_level[0], rho, size_lattice, cudaMemcpyDeviceToDevice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	error = EVALUATE_ERROR(N, N_block, error_block); 
	double temp, norm_in, error_in = 1.0;

	printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
	iter = 0;
	factor = 1.0;
	int dl;
	double one = 1.;
	double mone = -1.;
	double zero = 0.;
	
	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		dl = dimension_level[1];
		cudaMemset(field_level[1], 0, dl*dl*sizeof(double));
		RESTRICT_2D<<<bpg,tpb>>>(dl, r_level[0], r_level[1]);
		cublasDdot(handle, dl*dl, r_level[1], 1, r_level[1], 1, &error_in);		
//		printf("%.8e\n", error_in);
		norm_in = error_in;
		cudaMemcpy(p_level[1], r_level[1], dl*dl*sizeof(double), cudaMemcpyDeviceToDevice);
		while (sqrt(error_in)/norm_in>0.95)
		{
			LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor/4., p_level[1], A_p_level[1]);
	        cublasDdot(handle, dl*dl, p_level[1], 1, A_p_level[1], 1, &temp);
	        alpha = error_in/temp;
	        temp = -alpha;
	        cublasDaxpy(handle, dl*dl, &temp, A_p_level[1], 1, r_level[1], 1);
	        cublasDaxpy(handle, dl*dl, &alpha, p_level[1], 1, field_level[1], 1);
	        cublasDdot(handle, dl*dl, r_level[1], 1, r_level[1], 1, &temp);
	        beta = temp/error_in;
	        DAXPY<<<bpg,tpb>>>(dl, beta, p_level[1], r_level[1]);
	        error_in = temp;
//			printf("%.6e\n", error_in);
		}
  	
		dl = dimension_level[0];
		INTERPOLATE_2D<<<bpg,tpb>>>(dl, field_level[1], p_level[0]);
		cublasDaxpy(handle, dl*dl, &one, p_level[0], 1, field_level[0], 1);
		LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, field_level[0], r_level[0]);
		DAXPY<<<bpg,tpb>>>(dl, mone, r_level[0], rho);
		cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &error);		

		cudaMemcpy(p_level[0], r_level[0], dl*dl*sizeof(double), cudaMemcpyDeviceToDevice);
		LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, p_level[0], A_p_level[0]);
        cublasDdot(handle, dl*dl, p_level[0], 1, A_p_level[0], 1, &temp);
        alpha = error/temp;
        temp = -alpha;
        cublasDaxpy(handle, dl*dl, &temp, A_p_level[0], 1, r_level[0], 1);
        cublasDaxpy(handle, dl*dl, &alpha, p_level[0], 1, field_level[0], 1);
        cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &temp);
        beta = temp/error;
        DAXPY<<<bpg,tpb>>>(dl, beta, p_level[0], r_level[0]);
        error = temp;
		
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);

//		INTERPOLATE_2D<<<bpg,tpb>>>(dimension_level[0], field_level[1], p_level[0]);
//        cublasDaxpy(handle, dimension_level[0]*dimension_level[0], &one, p_level[0], 1, field_level[0], 1);
//        LAPLACIAN<<<bpg,tpb>>>(dimension_level[0], dx, photon_mass, factor, field_level[0], r_level[0]);
//        DAXPY<<<bpg,tpb>>>(dimension_level[0], mone, r_level[0], rho);
//        cublasDdot(handle, dimension_level[0]*dimension_level[0], r_level[0], 1, r_level[0], 1, &error);
//
//        cudaMemcpy(p_level[0], r_level[0], dimension_level[0]*dimension_level[0]*sizeof(double), cudaMemcpyDeviceToDevice);
//        LAPLACIAN<<<bpg,tpb>>>(dimension_level[0], dx, photon_mass, factor, p_level[0], A_p_level[0]);
//        cublasDdot(handle, dimension_level[0]*dimension_level[0], p_level[0], 1, A_p_level[0], 1, &temp);
//        alpha = error/temp;
//        temp = -alpha;
//        cublasDaxpy(handle, dimension_level[0]*dimension_level[0], &temp, A_p_level[0], 1, r_level[0], 1);
//        cublasDaxpy(handle, dimension_level[0]*dimension_level[0], &alpha, p_level[0], 1, field_level[0], 1);
//        cublasDdot(handle, dimension_level[0]*dimension_level[0], r_level[0], 1, r_level[0], 1, &temp);
//        beta = temp/error;
////      printf("%.4f\t%.4f\n", alpha, beta);
//        DAXPY<<<bpg,tpb>>>(dimension_level[0], beta, p_level[0], r_level[0]);
//        error = temp;
//
//        iter += 1;
//        if (iter%display_interval==0)
//            printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
	}
  
	output_field = fopen("simulated_field_distribution_GPU_TGCG.txt","w");
	FPRINTF(output_field, N, 1., field_level[0]);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

//	cudaFree(field);
//	cudaFree(r);
//	cudaFree(p);
//	cudaFree(A_p);
	cudaFree(field_analytic);
	cudaFree(rho);
	cudaFree(error_block);
	cudaFree(dimension_level);
	cublasDestroy(handle);
	fclose(output_field);
	fclose(output_rho);
	free(field_level);
	free(r_level);
	free(p_level);
	free(A_p_level);
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
