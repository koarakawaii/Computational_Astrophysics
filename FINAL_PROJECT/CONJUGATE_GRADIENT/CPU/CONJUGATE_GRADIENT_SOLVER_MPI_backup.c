#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>
#include <string.h>

void FPRINTF(FILE*, int, double, double*);
void INITIALIZE(int, double, double *, double *, double*, double*);
void LAPLACIAN(int, double, double, double*, double*);
void DAXPY(int, double, double*, double*);

int main(void)
{
	int N, N_per_rank, N_processors, display_interval;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria;
	double alpha, beta, error;
	long iter, iter_max;
	double *field, *rho, *r, *p, *A_p, *field_analytic;
	size_t size_lattice;
	FILE* output_field, *output_rho;
	printf("Solve the Poission problem using CG by MPI.\n\n");
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
	printf("Set the number of OpenMP processor. (N must be divisible by number of processors)\n");
	scanf("%d", &N_processors);
	if (N%N_processors!=0)
	{
		printf("N is not divisible by number of processors! Exit!\n");
		return EXIT_FAILURE;
	}
	else
	{
		printf("Number of OpenMP processors is %d .\n", N_processors);
		N_per_rank = N/N_processors;
	}
	printf("\n");

	printf("Start Preparation...\n");
	dx = 1./(N-1);	
	size_lattice = N*N*sizeof(double);
	output_field = fopen("analytical_field_distribution_CG_MPI.txt","w");
	output_rho = fopen("charge_distribution_CG.txt","w");

	double start = omp_get_wtime();
	field = malloc(size_lattice);
	r = malloc(size_lattice);
	p = malloc(size_lattice);
	A_p = malloc(size_lattice);
	field_analytic = malloc(size_lattice);
	rho = malloc(size_lattice);

	INITIALIZE(N, dx, rho, field, field_analytic, &error);
	memcpy(r, rho, size_lattice);
	memcpy(p, rho, size_lattice);
	
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);
	double end = omp_get_wtime();

	printf("Preparation ends.\n");
	preparation_time = end - start;
	printf("Total preparation time is %.4f s.\n\n", preparation_time);

	start = omp_get_wtime();
	double temp;

	printf("Starts computation with error = %.8e...\n", error);
	iter = 0;
	
	while (sqrt(error)/(double)(N-2)>criteria&&iter<iter_max)
	{
		LAPLACIAN(N, dx, photon_mass, p, A_p);
		temp = cblas_ddot(N*N, p, 1, A_p, 1);
		alpha = error/temp;
		temp = -alpha;
		cblas_daxpy(N*N, temp, A_p, 1, r, 1);
		cblas_daxpy(N*N, alpha, p, 1, field, 1);
		temp = cblas_ddot(N*N, r, 1, r, 1);
		beta = temp/error;
//		printf("%.4f\t%.4f\n", alpha, beta);
		DAXPY(N, beta, p, r);
		error = temp;
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/(double)(N-2));
	}
  
	output_field = fopen("simulated_field_distribution_MPI_CG.txt","w");
	FPRINTF(output_field, N, 1., field);
	end = omp_get_wtime();
	computation_time = end - start;
	printf("Computation time is %.4f s.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f s.\n", iter, total_time);

	free(field);
	free(r);
	free(p);
	free(field_analytic);
	free(rho);
	fclose(output_field);
	fclose(output_rho);
	return EXIT_SUCCESS;
}

void INITIALIZE(int N, double dx, double* rho, double* field, double* field_analytic, double *error)
{
	int idx_x, idx_y, L, R, U, D;
	double x, y;
	for (int idx=0; idx<N*N; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
	
		x = idx_x*dx;
		y = idx_y*dx;
	
		field_analytic[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			L = idx_x-1 + idx_y*N;
			R = idx_x+1 + idx_y*N;
			U = idx_x + (idx_y+1)*N;
			D = idx_x + (idx_y-1)*N;
			field[idx] = 0.0;
			rho[idx] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
			*error += pow((field[L]+field[R]+field[U]+field[D]-4.*field[idx])-rho[idx], 2.);
		}
		else
		{
			field[idx] = field_analytic[idx];
			rho[idx] = 0.0;
		}
	}
}

void LAPLACIAN(int N, double dx, double photon_mass, double* p, double* A_p)
{
	int idx_x, idx_y, L, R, U, D;
	
	for (int idx=0; idx<N*N; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			L = idx_x-1 + idx_y*N;
			R = idx_x+1 + idx_y*N;
			U = idx_x + (idx_y+1)*N;
			D = idx_x + (idx_y-1)*N;
	
			A_p[idx] = (p[L]+p[R]+p[U]+p[D]-(4.-pow(photon_mass*dx,2.))*p[idx]);
	//		printf("%d\t%.4f\n", idx, A_p[idx]);
		}
		else
			A_p[idx] = 0.0;
	}
}

void DAXPY(int N, double c, double *A, double *B)
{
	for (int idx=0; idx<N*N; idx++)
		A[idx] = c*A[idx] + B[idx];
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
