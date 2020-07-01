#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

void FPRINTF(FILE*, int, double, double*);
void INITIALIZE(int, double, double*, double*, double*);
double EVALUATE_ERROR(int, double*, double*);
void LAPLACIAN(int, double, double, double*, double*);
void DAXPY_X(int, double, double*, double*);	// x is replaced
void DAXPY_Y(int, double, double*, double*);	// y is replaced
double DDOT(int, double*, double*);

int main(void)
{
	int N, N_thread, display_interval;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria, error, norm;
	double alpha, beta;
	double start, end;
	long iter, iter_max;
	double *field, *rho, *r, *p, *A_p, *field_analytic;
	size_t size_lattice;
	FILE* output_field, *output_rho;

	printf("Solve the Poission problem using CG by OpenMP.\n\n");
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
	printf("Set the number of OpenMP threads.\n");
	scanf("%d", &N_thread);
	printf("%d OpenMP threads are set.\n", N_thread);
	printf("\n");
	printf("Start Preparation...\n");

	omp_set_num_threads(N_thread);
	start = omp_get_wtime();

	dx = 1./(N-1);	
	size_lattice = N*N*sizeof(double);
	field = malloc(size_lattice);
	field_analytic = malloc(size_lattice);
	rho = malloc(size_lattice);
	r = malloc(size_lattice);
	p = malloc(size_lattice);
	A_p = malloc(size_lattice);
	output_field = fopen("analytical_field_distribution_CG_OpenMP.txt","w");
	output_rho = fopen("charge_distribution_CG.txt","w");

	INITIALIZE(N, dx, rho, field_analytic, field);
	FPRINTF(output_field, N, 1., field_analytic);
	FPRINTF(output_rho, N, pow(dx,-2.), rho);
	fclose(output_field);

	norm = DDOT(N*N, rho, rho);
	norm = sqrt(norm);
	printf("Norm = %.4e .\n",norm);

	error = EVALUATE_ERROR(N, rho, field);
	memcpy(r, rho, size_lattice);
	memcpy(p, rho, size_lattice);
	
	end = omp_get_wtime();
	printf("Preparation ends.\n");
	preparation_time = end - start;
	printf("Total preparation time is %.4f s.\n\n", preparation_time);
	printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
	start = omp_get_wtime();

	double temp;
	iter = 0;
	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		LAPLACIAN(N, dx, photon_mass, p, A_p);
		temp = DDOT(N*N, p, A_p);
//		printf("%.4f\n", temp);
		alpha = error/temp;
		DAXPY_Y(N*N, -alpha, A_p, r);
		DAXPY_Y(N*N, alpha, p, field);
		temp = DDOT(N*N, r, r);
		beta = temp/error;
//		printf("%.4f\t%.4f\n", alpha, beta);
		DAXPY_X(N*N, beta, p, r);
		error = temp;
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
	}
  
	output_field = fopen("simulated_field_distribution_OpenMP_CG.txt","w");
	FPRINTF(output_field, N, 1., field);
	end = omp_get_wtime();
	computation_time = end - start;
	printf("Computation time is %.4f s.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iteration is %ld ; total time is %.4f s.\n", iter, total_time);

	free(field);
	free(field_analytic);
	free(rho);
	free(r);
	free(p);
	free(A_p);
	return EXIT_SUCCESS;
}

void INITIALIZE(int N, double dx, double* rho, double* field_analytic, double* field)
{
	double* temp = malloc(N*N*sizeof(double));

#	pragma omp parallel for
	for (int idx=0; idx<N*N; idx++)
	{
		int idx_x = idx%N;
		int idx_y = idx/N;
	
		double x = idx_x*dx;
		double y = idx_y*dx;
		temp[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
	
		field_analytic[idx] = temp[idx];
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			field[idx] = 0.0;
			rho[idx] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		}
		else
		{
			field[idx] = temp[idx];
			rho[idx] = 0.0;
		}
	}
	free(temp);
}

double EVALUATE_ERROR(int N, double* rho, double* field)
{
	double error = 0.0;
#	pragma omp parallel for reduction( +:error )
	for (int idx=0; idx<N*N; idx++)
	{
		int idx_x = idx%N;
		int idx_y = idx/N;
	
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			double L = field[idx-1];
			double R = field[idx+1];
			double U = field[idx+N];
			double D = field[idx-N];

			error += pow(L + R + U + D - 4.*field[idx] - rho[idx], 2.);
		}
	}
	return error;
}

void LAPLACIAN(int N, double dx, double photon_mass, double* p, double* A_p)
{
#	pragma omp parallel for 
	for (int idx=0; idx<N*N; idx++)
	{
		int idx_x = idx%N;
		int idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			double L = p[idx-1];
			double R = p[idx+1];
			double U = p[idx+N];
			double D = p[idx-N];
	
			A_p[idx] = ( L + R + U + D - (4.+pow(photon_mass*dx,2.))*p[idx]);
	//		printf("%d\t%.4f\n", idx, A_p[idx]);
		}
		else
			A_p[idx] = 0.0;
	}
}

void DAXPY_X(int N, double c, double *A, double *B)
{
#	pragma omp parallel for
	for (int idx=0; idx<N; idx++)
		A[idx] = c*A[idx] + B[idx];
}

void DAXPY_Y(int N, double c, double *A, double *B)
{
#	pragma omp parallel for
	for (int idx=0; idx<N; idx++)
		B[idx] = c*A[idx] + B[idx];
}

double DDOT(int N, double* A, double* B)
{
	double dot = 0.0;
#	pragma omp parallel for reduction( +:dot )
	for (int idx=0; idx<N; idx++)
		dot += A[idx]*B[idx];
//	printf("%.6f\n", dot);
	return dot;
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
