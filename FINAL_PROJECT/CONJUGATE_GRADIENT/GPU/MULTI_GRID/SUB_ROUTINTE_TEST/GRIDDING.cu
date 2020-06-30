#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

void PRINT_FIELD(int, double*);

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
//		printf("%d\t%.4e\t", idx_coarse, field_coarse[idx_coarse]);
	}
}

int main(void)
{
	int N, N_level;
	int tpb_x, tpb_y, bpg_x, bpg_y;
	int *dimension_level;
	double **field_level;
	printf("Test the interpolate and restrict for multi-grid by GPU.\n\n");
    printf("Enter the latttice size (N,N) .");
    scanf("%d", &N);
    printf("The lattice size is (%d,%d).\n", N, N);
//	printf("Set the depth of the V process level.\n");
//	scanf("%d", &N_level);
	printf("The depth of the V process level will be set automatically.\n");
	N_level = (int)(log2((N-1)/4.));
	printf("The depth of the V process is %d .\n", N_level);
//    printf("Set the photon mass.\n");
//    scanf("%lf", &photon_mass);
//    printf("The photon mass is %.4e .\n", photon_mass);
    printf("Set the GPU threads per block (tx,ty). \n");
    scanf("%d %d", &tpb_x, &tpb_y);
    printf("Threads per block for GPU is (%d,%d) .\n", tpb_x, tpb_y);
    printf("The block per grid will be set automatically.");
    bpg_x = (N+tpb_x-1)/tpb_x;
    bpg_y = (N+tpb_y-1)/tpb_y;
    printf("Blocks per grid for GPU is (%d,%d) .\n", bpg_x, bpg_y);
    printf("\n");
	
	cudaSetDevice(0);
	dim3 tpb(tpb_x,tpb_y);
	dim3 bpg(bpg_x,bpg_y);
	cudaMallocManaged(&dimension_level, (N_level+1)*sizeof(int));

	field_level = (double**)malloc((N_level+1)*sizeof(double*));
	int dimension = N-1;
	for (int level=0; level<=N_level; level++)
	{
		cudaMallocManaged(&field_level[level], (dimension+1)*(dimension+1)*sizeof(double));
		dimension_level[level] = dimension + 1;
		dimension /= 2;
	}

	for (int i=0; i<dimension_level[0]*dimension_level[0]; i++)
//		field_level[0][i] = 1.0;
		field_level[0][i] = i;

//	RESTRICT_2D<<<bpg,tpb>>>(dimension_level[1], field_level[0], field_level[1]);
//	INTERPOLATE_2D<<<bpg,tpb>>>(dimension_level[0], field_level[1], field_level[0]);
//	cudaDeviceSynchronize();

	for (int i=0; i<N_level; i++)
	{
		RESTRICT_2D<<<bpg,tpb>>>(dimension_level[i+1], field_level[i], field_level[i+1]);
		cudaDeviceSynchronize();
	}

	for (int j=0; j<N_level; j++)
	{
		for (int i=0; i<dimension_level[j]*dimension_level[j]; i++)
			field_level[j][i] = 0.0;
	}

	for (int i=N_level; i>=1; i--)
	{
		INTERPOLATE_2D<<<bpg,tpb>>>(dimension_level[i-1], field_level[i], field_level[i-1]);
		cudaDeviceSynchronize();
	}
	
//	PRINT_FIELD(dimension_level[1], field_level[1]);
	PRINT_FIELD(dimension_level[0], field_level[0]);

	free(field_level);
	cudaFree(dimension_level);
	return EXIT_SUCCESS;
}

void PRINT_FIELD(int dimension, double* field)
{
	for (int j=0; j<dimension*dimension; j++)
//		printf("%.4e\n", field[j]);
		printf("%.2f\n", field[j]);
}
