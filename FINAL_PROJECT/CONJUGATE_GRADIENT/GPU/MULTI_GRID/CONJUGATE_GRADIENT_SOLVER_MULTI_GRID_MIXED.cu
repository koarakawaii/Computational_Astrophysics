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

__global__ void LAPLACIAN_DOUBLE(int N, double dx, double photon_mass, double factor, double* p, double* A_p)
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
		}
		else
			A_p[idx] = factor*p[idx];
	}
}

__global__ void LAPLACIAN_FLOAT(int N, float dx, float photon_mass, float factor, float* p, float* A_p)
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
	
			A_p[idx] = factor*(p[L]+p[R]+p[U]+p[D]-(4.+powf(photon_mass*dx,2.))*p[idx]);
		}
		else
			A_p[idx] = factor*p[idx];
	}
}

__global__ void DAXPY_DOUBLE(int N, double c, double *A, double *B)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + N*idx_y;
	
		A[idx] = c*A[idx] + B[idx];
	}
}

__global__ void DAXPY_FLOAT(int N, float c, float *A, float *B)
{
	int idx_x = threadIdx.x + blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y*blockIdx.y;
	if (idx_x<N&&idx_y<N)
	{
		int idx = idx_x + N*idx_y;
	
		A[idx] = c*A[idx] + B[idx];
	}
}

__global__ void INTERPOLATE_2D_LAST(int dimension, float* field_coarse, double* field_fine)
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
			field_fine[idx_fine] = (double)(field_coarse[idx_coarse]);
		else if (idx_x_fine%2==1&&idx_y_fine%2==0)
			field_fine[idx_fine] = (double)(0.5*(field_coarse[idx_coarse]+field_coarse[idx_coarse+1]));
		else if (idx_x_fine%2==0&&idx_y_fine%2==1)
			field_fine[idx_fine] = (double)(0.5*(field_coarse[idx_coarse]+field_coarse[idx_coarse+N_coarse]));
		else
			field_fine[idx_fine] = (double)(0.25*(field_coarse[idx_coarse]+field_coarse[idx_coarse+1]+field_coarse[idx_coarse+N_coarse]+field_coarse[idx_coarse+N_coarse+1]));
	}
}

__global__ void INTERPOLATE_2D(int dimension, float* field_coarse, float* field_fine)
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

__global__ void RESTRICT_2D_FIRST(int dimension, double* field_fine, float* field_coarse)
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
			field_coarse[idx_coarse] = (float)(1./16.*(field_fine[idx_fine-4]+field_fine[idx_fine-2]+field_fine[idx_fine+2]+field_fine[idx_fine+4]) + 1./8.*(field_fine[idx_fine-3]+field_fine[idx_fine-1]+field_fine[idx_fine+1]+field_fine[idx_fine+3]) + 1./4.*field_fine[idx_fine]);
		else 
			field_coarse[idx_coarse] = (float)(field_fine[idx_fine]);
	}
}

__global__ void RESTRICT_2D(int dimension, float* field_fine, float* field_coarse)
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
			field_coarse[idx_coarse] = (float)(1./16.*(field_fine[idx_fine-4]+field_fine[idx_fine-2]+field_fine[idx_fine+2]+field_fine[idx_fine+4]) + 1./8.*(field_fine[idx_fine-3]+field_fine[idx_fine-1]+field_fine[idx_fine+1]+field_fine[idx_fine+3]) + 1./4.*field_fine[idx_fine]);
		else 
			field_coarse[idx_coarse] = (float)(field_fine[idx_fine]);
	}
}

int main(void)
{
	int N, N_level, inner_loop, N_block, display_interval, tpb_x, tpb_y, bpg_x, bpg_y;
	float preparation_time, computation_time, total_time;
	float alpha_f, beta_f;
	double photon_mass, dx, criteria;
	double alpha_d, beta_d, error;
	long iter, iter_max;
    int *dimension_level;
	float *p_f, *A_p_f, *r_temp_f;
    float **field_level_f, **r_level_f;
	double *rho, *p_d, *A_p_d, *field_analytic, *error_block, *r_temp_d;
	double *field_level_d, *r_level_d; 
	size_t size_lattice_d, size_lattice_f, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;

	printf("Solve the Poission problem using CG by GPU.\n\n");
	printf("Enter the latttice size (N,N) .");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("Set the depth of the V process.\n");
	scanf("%d",&N_level);
	printf("The depth of V process is %d .\n", N_level);
	printf("Set the number of inner-loop.\n");
	scanf("%d",&inner_loop);
	printf("The number of inner-loop is %d .\n", inner_loop);
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
	size_lattice_d = N*N*sizeof(double);
	size_lattice_f = N*N*sizeof(float);
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
	cudaMallocManaged(&p_f, size_lattice_f);
	cudaMallocManaged(&A_p_f, size_lattice_f);
	cudaMallocManaged(&p_d, size_lattice_d);
	cudaMallocManaged(&A_p_d, size_lattice_d);
	cudaMallocManaged(&field_analytic, size_lattice_d);
	cudaMallocManaged(&rho, size_lattice_d);
	cudaMallocManaged(&error_block, N_block*sizeof(double));
	cudaMallocManaged(&dimension_level, (N_level+1)*sizeof(int));
	cudaMallocManaged(&r_temp_f, size_lattice_f);
	cudaMallocManaged(&r_temp_d, size_lattice_d);
	cudaMallocManaged(&field_level_d, size_lattice_d);
	cudaMallocManaged(&r_level_d, size_lattice_d);

	/* allocate the memory for multi-grid */
    field_level_f = (float**)malloc(N_level*sizeof(float*));
    r_level_f = (float**)malloc(N_level*sizeof(float*));
    int dimension = (N-1)/2;
    for (int level=0; level<N_level; level++)
    {
        cudaMallocManaged(&field_level_f[level], (dimension+1)*(dimension+1)*sizeof(float));
        cudaMallocManaged(&r_level_f[level], (dimension+1)*(dimension+1)*sizeof(float));
        dimension_level[level] = dimension + 1;
        dimension /= 2;
    }
	INITIALIZE<<<bpg,tpb>>>(N, dx, rho, field_level_d, field_analytic);
	EVALUATE_ERROR_BLOCK<<<bpg,tpb,size_sm>>>(N, rho, field_level_d, error_block);
	double norm;
	cublasDdot(handle, N*N, rho, 1, rho, 1, &norm);
	norm = sqrt(norm);
	
	cudaMemcpy(r_level_d, rho, size_lattice_d, cudaMemcpyDeviceToDevice);
	if (N_level==0)
		cudaMemcpy(p_d, rho, size_lattice_d, cudaMemcpyDeviceToDevice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	error = EVALUATE_ERROR(N, N_block, error_block); 
	float temp_f, norm_in, error_in = 1.0;
	double temp_d;

	printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
	iter = 0;
	int dl;
	float one_f = 1.;
	float mone_f = -1.;
	float zero_f = 0.;
	double one_d = 1.;
	double mone_d = -1.;
	double zero_d = 0.;
	
//	cudaMemcpy(p_d, rho, size_lattice_d, cudaMemcpyDeviceToDevice);

	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		if (N_level!=0)
		{
			RESTRICT_2D_FIRST<<<bpg,tpb>>>(dimension_level[0], r_level_d, r_level_f[0]);
			for (int l=0; l<N_level-1; l++)
			{
				dl = dimension_level[l+1];
  			RESTRICT_2D<<<bpg,tpb>>>(dl, r_level_f[l], r_level_f[l+1]);
  		}
  
			for (int l=N_level-1; l>=0; l--)
			{
				dl = dimension_level[l];
				cudaMemset(field_level_f[l], 0, dl*dl*sizeof(float));
				if (l<N_level-1)
				{
					cudaMemcpy(r_temp_f, r_level_f[l], dl*dl*sizeof(float), cudaMemcpyDeviceToDevice);
					INTERPOLATE_2D<<<bpg,tpb>>>(dl, field_level_f[l+1], p_f);
					cublasSaxpy(handle, dl*dl, &one_f, p_f, 1, field_level_f[l], 1);
					LAPLACIAN_FLOAT<<<bpg,tpb>>>(dl, (float)(dx), (float)(photon_mass), 1./powf(2.,2.*l), field_level_f[l], r_level_f[l]);
					DAXPY_FLOAT<<<bpg,tpb>>>(dl, mone_f, r_level_f[l], r_temp_f);
				}
				cublasSdot(handle, dl*dl, r_level_f[l], 1, r_level_f[l], 1, &error_in);		
				norm_in = error_in;
				cudaMemcpy(p_f, r_level_f[l], dl*dl*sizeof(float), cudaMemcpyDeviceToDevice);
	
				for (int loop=0; loop<inner_loop; loop++)
				{
					LAPLACIAN_FLOAT<<<bpg,tpb>>>(dl, (float)(dx), (float)(photon_mass), 1./powf(2.,2.*l), p_f, A_p_f);
			        cublasSdot(handle, dl*dl, p_f, 1, A_p_f, 1, &temp_f);
			        alpha_f = error_in/temp_f;
			        temp_f = -alpha_f;
			        cublasSaxpy(handle, dl*dl, &temp_f, A_p_f, 1, r_level_f[l], 1);
			        cublasSaxpy(handle, dl*dl, &alpha_f, p_f, 1, field_level_f[l], 1);
			        cublasSdot(handle, dl*dl, r_level_f[l], 1, r_level_f[l], 1, &temp_f);
			        beta_f = temp_f/error_in;
			        DAXPY_FLOAT<<<bpg,tpb>>>(dl, beta_f, p_f, r_level_f[l]);
			        error_in = temp_f;
				}
			}
	        INTERPOLATE_2D_LAST<<<bpg,tpb>>>(N, field_level_f[0], p_d);
	        cublasDaxpy(handle, N*N, &one_d, p_d, 1, field_level_d, 1);
	        LAPLACIAN_DOUBLE<<<bpg,tpb>>>(N, dx, photon_mass, 1., field_level_d, r_level_d);
	        DAXPY_DOUBLE<<<bpg,tpb>>>(N, mone_d, r_level_d, rho);
	        cublasDdot(handle, N*N, r_level_d, 1, r_level_d, 1, &error);
			cudaMemcpy(p_d, r_level_d, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
		}
		else
			dl = N;

        LAPLACIAN_DOUBLE<<<bpg,tpb>>>(N, dx, photon_mass, 1., p_d, A_p_d);
        cublasDdot(handle, N*N, p_d, 1, A_p_d, 1, &temp_d);
        alpha_d = error/temp_d;
        temp_d = -alpha_d;
        cublasDaxpy(handle, N*N, &temp_d, A_p_d, 1, r_level_d, 1);
        cublasDaxpy(handle, N*N, &alpha_d, p_d, 1, field_level_d, 1);
        cublasDdot(handle, N*N, r_level_d, 1, r_level_d, 1, &temp_d);
        beta_d = temp_d/error;
        DAXPY_DOUBLE<<<bpg,tpb>>>(N, beta_d, p_d, r_level_d);
        error = temp_d;

        iter += 1;
        if (iter%display_interval==0)
            printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
	}
  
	output_field = fopen("simulated_field_distribution_GPU_MGCG_MIXED.txt","w");
	FPRINTF(output_field, N, 1., field_level_d);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

	cudaFree(p_d);
	cudaFree(A_p_d);
	cudaFree(p_f);
	cudaFree(A_p_f);
	cudaFree(field_analytic);
	cudaFree(rho);
	cudaFree(error_block);
	cudaFree(dimension_level);
	cudaFree(r_temp_f);
	cudaFree(r_temp_d);
	cudaFree(field_level_d);
	cudaFree(r_level_d);
	cublasDestroy(handle);
	fclose(output_field);
	fclose(output_rho);
	free(field_level_f);
	free(r_level_f);
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
