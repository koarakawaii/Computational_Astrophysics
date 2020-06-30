#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define FINEST_GRID 17

double EVALUATE_ERROR(int, int, double*);
double DOT_LU(int, double*, double*);
void MATRIX_MULTIPLY_LU(int, int, double**, double**);
void INVERSE_MATRIX(int, double*, double*);
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
			A_p[idx] = 0.0;
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
	int N, N_level, N_block, inner_loop, display_interval, tpb_x, tpb_y, bpg_x, bpg_y, row;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria;
	double alpha, beta, error, factor;
	long iter, iter_max;
    int *dimension_level;
//	double *rho, *field_analytic, *error_block;
	double *rho, *p, *A_p, *field_analytic, *error_block;
	double *A, *A_inverse;
//    double **field_level, **r_level, **p_level, **A_p_level;
    double **field_level, **r_level;
	size_t size_lattice, size_sm;
	cudaEvent_t start, stop;
	FILE* output_field, *output_rho;

	printf("Solve the Poission problem using CG by GPU.\n\n");
	printf("Enter the latttice size (N,N) .");
	scanf("%d", &N);
	printf("The lattice size is (%d,%d).\n", N, N);
	printf("The depth of the V process level will be set automatically.\n");
    N_level = (int)(log2((N-1)/(FINEST_GRID-1)));
    printf("The depth of the V process is %d .\n", N_level);
	printf("Set the photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is %.4e .\n", photon_mass);
	printf("Set the maximum iteration times.\n");
	scanf("%ld", &iter_max);
	printf("The maximum iteration times is %ld .\n", iter_max);
	printf("Set the stopping criteria.\n");
	scanf("%lf", &criteria);
	printf("The stopping criteria is %.4e .\n", criteria);
	printf("Set the number of inner loops.\n");
	scanf("%d", &inner_loop);
	printf("Number of inner loop is %d .\n", inner_loop);
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
	row = FINEST_GRID*FINEST_GRID;
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
	cudaMallocManaged(&p, size_lattice);
	cudaMallocManaged(&A_p, size_lattice);
	cudaMallocManaged(&field_analytic, size_lattice);
	cudaMallocManaged(&rho, size_lattice);
	cudaMallocManaged(&error_block, N_block*sizeof(double));
	cudaMallocManaged(&dimension_level, (N_level+1)*sizeof(int));
	cudaMallocManaged(&A_inverse, row*row*sizeof(double));

	/* allocate the memory for multi-grid */
    field_level = (double**)malloc((N_level+1)*sizeof(double*));
    r_level = (double**)malloc((N_level+1)*sizeof(double*));
//    p_level = (double**)malloc((N_level+1)*sizeof(double*));
//    A_p_level = (double**)malloc((N_level+1)*sizeof(double*));
    int dimension = N-1;
    for (int level=0; level<=N_level; level++)
    {
        cudaMallocManaged(&field_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        cudaMallocManaged(&r_level[level], (dimension+1)*(dimension+1)*sizeof(double));
//        cudaMallocManaged(&p_level[level], (dimension+1)*(dimension+1)*sizeof(double));
//        cudaMallocManaged(&A_p_level[level], (dimension+1)*(dimension+1)*sizeof(double));
        dimension_level[level] = dimension + 1;
        dimension /= 2;
    }

	/* get inverse of discret Laplacian */
    A = (double*)calloc(row*row,sizeof(double));
	factor = 1.0/pow(2.,2*N_level);
//    A_inverse = (double*)calloc(row*row,sizeof(double));
    for (int i=0; i<row; i++)
    {
        int i_x = i%FINEST_GRID;
        int i_y = i/FINEST_GRID;

        if (i_x!=0&&i_x!=FINEST_GRID-1&&i_y!=0&&i_y!=FINEST_GRID-1)
        {
            if (i_x>1)
                A[(i-1) + i*row] = factor;
            if (i_x<FINEST_GRID-2)
                A[(i+1) + i*row] = factor;
            if (i_y>1)
                A[(i-FINEST_GRID) + i*row] = factor;
            if (i_y<FINEST_GRID-2)
                A[(i+FINEST_GRID) + i*row] = factor;
            A[i*(row + 1)] = -factor*(4. - pow(photon_mass*dx,2.));
        }
        else
            A[i*(row + 1)] = factor;
    }
    INVERSE_MATRIX(row, A, A_inverse);
	
	INITIALIZE<<<bpg,tpb>>>(N, dx, rho, field_level[0], field_analytic);
	EVALUATE_ERROR_BLOCK<<<bpg,tpb,size_sm>>>(N, rho, field_level[0], error_block);
	double norm;
	cublasDdot(handle, N*N, rho, 1, rho, 1, &norm);
	norm = sqrt(norm);
	
	cudaDeviceSynchronize();
	cudaMemcpy(r_level[0], rho, size_lattice, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(p_level[0], rho, size_lattice, cudaMemcpyDeviceToDevice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);

	printf("Preparation ends.\n");
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&preparation_time, start, stop);
	printf("Total preparation time is %.4f ms.\n\n", preparation_time);

	cudaEventRecord(start,0);	
	error = EVALUATE_ERROR(N, N_block, error_block); 
	double temp;

	printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
	iter = 0;
	factor = 1.0;
	int dl;
	double one = 1.;
	double mone = -1.;
	double zero = 0.;

	for (int level=0; level<N_level; level++)
	{
		dl = dimension_level[level+1];
		RESTRICT_2D<<<bpg,tpb>>>(dl, r_level[level], r_level[level+1]);
	}

	cublasDgemv(handle, CUBLAS_OP_N, row, row, &one, A_inverse, row, r_level[N_level], 1, &zero, field_level[N_level], 1);
	for (int level=N_level; level>=1; level--)
	{
		dl = dimension_level[level-1];
		INTERPOLATE_2D<<<bpg,tpb>>>(dl, field_level[level], p);
	}

	dl = dimension_level[0];
	cublasDaxpy(handle, dl*dl, &one, p, 1, field_level[0], 1);
	LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, field_level[0], r_level[0]);
	DAXPY<<<bpg,tpb>>>(dl, mone, r_level[0], rho);
	cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &error);		
	cudaMemcpy(p, r_level[0], dl*dl*sizeof(double), cudaMemcpyDeviceToDevice);
	
	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
//		for (int level=0; level<N_level; level++)
//			RESTRICT_2D<<<bpg,tpb>>>(dimension_level[level+1], r_level[level], r_level[level+1]);
//	
//		cublasDgemv(handle, CUBLAS_OP_N, row, row, &one, A_inverse, row, r_level[N_level], 1, &zero, field_level[N_level], 1);
//		for (int level=N_level; level>=1; level--)
//			INTERPOLATE_2D<<<bpg,tpb>>>(dimension_level[level-1], field_level[level], p_level[level-1]);
//
//		cublasDaxpy(handle, dimension_level[0]*dimension_level[0], &one, p_level[0], 1, field_level[0], 1);
//		LAPLACIAN<<<bpg,tpb>>>(dimension_level[0], dx, photon_mass, factor, field_level[0], r_level[0]);
//		DAXPY<<<bpg,tpb>>>(dimension_level[0], mone, r_level[0], rho);
//		cublasDdot(handle, dimension_level[0]*dimension_level[0], r_level[0], 1, r_level[0], 1, &error);		
//
//		cudaMemcpy(p_level[0], r_level[0], dimension_level[0]*dimension_level[0]*sizeof(double), cudaMemcpyDeviceToDevice);
//		LAPLACIAN<<<bpg,tpb>>>(dimension_level[0], dx, photon_mass, factor, p_level[0], A_p_level[0]);
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
		
//		int dl;
//		for (int level=0; level<N_level; level++)
//		{
//			dl = dimension_level[level+1];
//			RESTRICT_2D<<<bpg,tpb>>>(dl, r_level[level], r_level[level+1]);
//		}
//	
//		cublasDgemv(handle, CUBLAS_OP_N, row, row, &one, A_inverse, row, r_level[N_level], 1, &zero, field_level[N_level], 1);
//		for (int level=N_level; level>=1; level--)
//		{
//			dl = dimension_level[level-1];
//			INTERPOLATE_2D<<<bpg,tpb>>>(dl, field_level[level], p);
//		}
//
//		dl = dimension_level[0];
//		cublasDaxpy(handle, dl*dl, &one, p, 1, field_level[0], 1);
//		LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, field_level[0], r_level[0]);
//		DAXPY<<<bpg,tpb>>>(dl, mone, r_level[0], rho);
//		cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &error);		
//		cudaMemcpy(p, r_level[0], dl*dl*sizeof(double), cudaMemcpyDeviceToDevice);
		
//		for (int loop=0; loop<inner_loop; loop++)
//		{
			LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, p, A_p);
	        cublasDdot(handle, dl*dl, p, 1, A_p, 1, &temp);
	        alpha = error/temp;
	        temp = -alpha;
	        cublasDaxpy(handle, dl*dl, &temp, A_p, 1, r_level[0], 1);
	        cublasDaxpy(handle, dl*dl, &alpha, p, 1, field_level[0], 1);
	        cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &temp);
	        beta = temp/error;
//			printf("%.4f\t%.4f\n", alpha, beta);
	        DAXPY<<<bpg,tpb>>>(dl, beta, p, r_level[0]);
	        error = temp;
			iter += 1;
			if (iter%display_interval==0)
				printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
			if (iter%inner_loop==0)
			{
				for (int level=0; level<N_level; level++)
				{
					dl = dimension_level[level+1];
					RESTRICT_2D<<<bpg,tpb>>>(dl, r_level[level], r_level[level+1]);
				}
			
				cublasDgemv(handle, CUBLAS_OP_N, row, row, &one, A_inverse, row, r_level[N_level], 1, &zero, field_level[N_level], 1);
				for (int level=N_level; level>=1; level--)
				{
					dl = dimension_level[level-1];
					INTERPOLATE_2D<<<bpg,tpb>>>(dl, field_level[level], p);
				}
			
				dl = dimension_level[0];
				cublasDaxpy(handle, dl*dl, &one, p, 1, field_level[0], 1);
				LAPLACIAN<<<bpg,tpb>>>(dl, dx, photon_mass, factor, field_level[0], r_level[0]);
				DAXPY<<<bpg,tpb>>>(dl, mone, r_level[0], rho);
				cublasDdot(handle, dl*dl, r_level[0], 1, r_level[0], 1, &error);		
				cudaMemcpy(p, r_level[0], dl*dl*sizeof(double), cudaMemcpyDeviceToDevice);
			}
//		}
	}
  
	output_field = fopen("simulated_field_distribution_GPU_MGCG.txt","w");
	FPRINTF(output_field, N, 1., field_level[0]);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	printf("Computation time is %.4f ms.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f ms.\n", iter, total_time);

//	cudaFree(field);
//	cudaFree(r);
	cudaFree(p);
	cudaFree(A_p);
	cudaFree(field_analytic);
	cudaFree(rho);
	cudaFree(error_block);
	cudaFree(dimension_level);
	cudaFree(A_inverse);
	cublasDestroy(handle);
	fclose(output_field);
	fclose(output_rho);
	free(field_level);
	free(r_level);
//	free(p_level);
//	free(A_p_level);
    free(A);
//    free(A_inverse);
	return EXIT_SUCCESS;
}

double EVALUATE_ERROR(int N, int N_block, double* error_block)
{
	double error = 0.0;
	for (int i=0; i<N_block; i++)
		error += error_block[i];
	return error;
}

double DOT_LU(int dim, double *a, double *b)
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

	int pivot_index;
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
		temp_L = (double*)malloc(j*sizeof(double));
		L[j][j] = A[j*(N_col+1)] - DOT_LU(j, L[j], U[j]);
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
			L[j][j] = A[j*(N_col+1)] - DOT_LU(j, L[j], U[j]);
		}
		
		for (int i=j+1; i<N_row; i++)
			L[i][j] = A[i*N_col+j] - DOT_LU(j, L[i], U[j]);
		for (int i=j+1; i<N_col; i++)
			U[i][j] = (A[j*N_col+i] - DOT_LU(j, L[j], U[i]))/L[j][j];
	}
	double *b;
	for (int j=0; j<N_col; j++)
	{
		b = (double*)calloc(N_row, sizeof(double));
		b[j] = 1.;
		A_inverse[j*N_col] = b[0]/L[0][0];
		for (int i=1; i<N_row; i++)
			A_inverse[j*N_col+i] = (b[i] - DOT_LU(i, L[i], A_inverse+j*N_col))/L[i][i];
		for (int i=N_row-2; i>=0; i--)
		{
			int count = N_row-1-i;
			temp_U = (double*)malloc(count*sizeof(double));
			for (int k=0; k<count; k++)
				temp_U[k] = U[i+1+k][i];
			A_inverse[j*N_col+i] -= DOT_LU(count, temp_U, A_inverse+j*N_col+i+1);
		}
	}
	free(L);
	free(U);
	free(temp_A);
	free(temp_L);
	free(temp_U);
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
