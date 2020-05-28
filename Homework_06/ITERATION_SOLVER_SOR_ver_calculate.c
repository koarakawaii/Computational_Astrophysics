/* This version uses direct calculation to obtain the neighor indices of even sites and odd sites.*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double EVALUATE_ERROR(int, double, double*, double*, double*, double*);
void LAPLACIAN_SOR(int, double, double, double*, double*, double*, double*);
void FPRINTF(FILE*, int N, double*);

int main(void)
{
	int N, N_threads, display_interval;
	float preparation_time, computation_time, total_time;
	double omega, dx, criteria;
	long iter, iter_max;
	double *field_even, *field_odd, *rho_even, *rho_odd, *field_final, *field_analytic, *rho;
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
	printf("Set the number of OpenMP threads.\n");
	scanf("%d", &N_threads);
	printf("The number of OpenMP threads is %d .\n", N_threads);
	printf("\n");

	double start = omp_get_wtime();	
	printf("Start Preparation...\n");
	dx = 1./(N-1);
	field_even = calloc((N*N)/2,sizeof(double));
	field_odd = calloc((N*N)/2,sizeof(double));
	field_final = malloc(N*N*sizeof(double));
	field_analytic = malloc(N*N*sizeof(double));
	rho_even = malloc((N*N)/2*sizeof(double));
	rho_odd = malloc((N*N)/2*sizeof(double));
	rho = malloc(N*N*sizeof(double));
	output_field = fopen("analytical_field_distribution.txt","w");
	output_rho = fopen("charge_distribution.txt","w");

	omp_set_num_threads(N_threads);
#	pragma omp parallel for
		for (int i_E=0; i_E<(N*N)/2; i_E++)
		{
			int ix = (2*i_E)%N;		
			int iy = (2*i_E)/N;
			int parity = (ix+iy)%2;
			ix += parity;
			double x = ix*dx;
			double y = iy*dx;
			rho_even[i_E] = 2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y);
			rho[ix+iy*N] = rho_even[i_E];
			field_analytic[ix+iy*N] = x*(1.-x)*y*(1.-y)*exp(x-y);
			if ( ix==0 || ix==N-1 || iy==0 || iy==N-1 )
				field_even[i_E] = field_analytic[ix+iy*N];
		}
#	pragma omp parallel for
		for (int i_O=0; i_O<(N*N)/2; i_O++)
		{
			int ix = (2*i_O)%N;		
			int iy = (2*i_O)/N;
			int parity = (ix+iy+1)%2;
			ix += parity;
			double x = ix*dx;
			double y = iy*dx;
			rho_odd[i_O] = 2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y);
			rho[ix+iy*N] = rho_odd[i_O];
			field_analytic[ix+iy*N] = x*(1.-x)*y*(1.-y)*exp(x-y);
			if ( ix==0 || ix==N-1 || iy==0 || iy==N-1 )
				field_odd[i_O] = field_analytic[ix+iy*N];
		}
	FPRINTF(output_field, N, field_analytic);
	FPRINTF(output_rho, N, rho);
	double end = omp_get_wtime();	
	printf("Preparation ends\n");
	preparation_time = end-start;
	printf("Total preparation time is %.4f s.\n\n", preparation_time);

	start = omp_get_wtime();	
	double error = EVALUATE_ERROR(N, dx, field_even, field_odd, rho_even, rho_odd);
	iter = 0;
	printf("Starts computation with error = %.8e...\n", error);
	while (error>criteria&&iter<iter_max)
	{
		LAPLACIAN_SOR(N, dx, omega, field_even, field_odd, rho_even, rho_odd);
		error = EVALUATE_ERROR(N, dx, field_even, field_odd, rho_even, rho_odd);
		iter += 1;
		if (iter%display_interval==0)
			printf("Iteration = %ld , error = %.8e .\n", iter, error);
	}

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
	
//	for (int i=0; i<(N*N)/2; i++) 
//		printf("i = %d, field componenet = %.4f .\n",i, field_even[i]);
//	for (int j=0; j<N; j++)
//	{
//		for (int i=0; i<N; i++)
//			fprintf(output_field, "%.8e\t", field_analytic[i+j*N]);
//		fprintf(output_field, "\n");
//	}
	output_field = fopen("simulated_field_distribution_calculate.txt","w");
	FPRINTF(output_field, N, field_final);
	end = omp_get_wtime();	
	computation_time = end-start;
	printf("Computation time is %.4f s.\n", computation_time);
	total_time = preparation_time + computation_time;
	printf("Total iterations is %ld ; total time is %.4f s.\n", iter, total_time);

	free(field_even);
	free(field_odd);
	free(rho_even);
	free(rho_odd);
	free(rho);
	free(field_analytic);
	free(field_final);
	fclose(output_field);
	fclose(output_rho);
	return EXIT_SUCCESS;
}

double EVALUATE_ERROR(int N, double dx, double* field_even, double* field_odd, double* rho_even, double* rho_odd)
{
	double error = 0.0;
#	pragma omp parallel for reduction(+:error)
	for (int i_E=N/2; i_E<(N*N)/2-N/2; i_E++)
	{
		int i_x = i_E%(N/2);
		int i_y = i_E/(N/2);
		if (i_y%2==0)
		{
			if (i_x!=0)
			{
				int L = i_x-1 + i_y*(N/2);
				int R = i_E;	
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				error += pow(( field_odd[L] + field_odd[R] + field_odd[D] + field_odd[U] - 4.*field_even[i_E])/(dx*dx)-rho_even[i_E], 2.);
//				printf("%d\n", i_E);
			}
		}		
		else
		{
			if (i_x!=(N/2)-1)
			{
				int L = i_E;
				int R = i_x+1 + i_y*(N/2);	
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				error += pow(( field_odd[L] + field_odd[R] + field_odd[D] + field_odd[U] - 4.*field_even[i_E])/(dx*dx)-rho_even[i_E], 2.);
//				printf("%d\n", i_E);
			}
		}
	}
#	pragma omp parallel for reduction(+:error)
	for (int i_O=N/2; i_O<(N*N)/2-N/2; i_O++)
	{
		int i_x = i_O%(N/2);
		int i_y = i_O/(N/2);
		if (i_y%2==0)
		{
			if (i_x!=(N/2)-1)
			{
				int L = i_O;	
				int R = i_x+1 + i_y*(N/2);
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				error += pow(( field_even[L] + field_even[R] + field_even[D] + field_even[U] - 4.*field_odd[i_O])/(dx*dx)-rho_odd[i_O], 2.);
//				printf("%d\n", i_O);
			}
		}		
		else
		{
			if (i_x!=0)
			{
				int L = i_x-1 + i_y*(N/2);	
				int R = i_O;
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				error += pow(( field_even[L] + field_even[R] + field_even[D] + field_even[U] - 4.*field_odd[i_O])/(dx*dx)-rho_odd[i_O], 2.);
//				printf("%d\n", i_O);
			}
		}
	}
	return sqrt(error/(double)((N-2)*(N-2)));
}

void LAPLACIAN_SOR(int N, double dx, double omega, double* field_even, double* field_odd, double *rho_even, double *rho_odd)
{
#	pragma omp parallel for
	for (int i_E=N/2; i_E<(N*N)/2-N/2; i_E++)
	{
		int i_x = i_E%(N/2);
		int i_y = i_E/(N/2);
		if (i_y%2==0)
		{
			if (i_x!=0)
			{
				int L = i_x-1 + i_y*(N/2);
				int R = i_E;	
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				field_even[i_E] += 0.25*omega*( field_odd[L] + field_odd[R] + field_odd[D] + field_odd[U] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
			}
		}		
		else
		{
			if (i_x!=(N/2)-1)
			{
				int L = i_E;
				int R = i_x+1 + i_y*(N/2);	
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				field_even[i_E] += 0.25*omega*( field_odd[L] + field_odd[R] + field_odd[D] + field_odd[U] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
			}
		}
	}
#	pragma omp parallel for
	for (int i_O=N/2; i_O<(N*N)/2-N/2; i_O++)
	{
		int i_x = i_O%(N/2);
		int i_y = i_O/(N/2);
		if (i_y%2==0)
		{
			if (i_x!=(N/2)-1)
			{
				int L = i_O;	
				int R = i_x+1 + i_y*(N/2);
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				field_odd[i_O] += 0.25*omega*( field_even[L] + field_even[R] + field_even[D] + field_even[U] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
			}
		}		
		else
		{
			if (i_x!=0)
			{
				int L = i_x-1 + i_y*(N/2);	
				int R = i_O;
				int D = i_x + (i_y-1)*(N/2);
				int U = i_x + (i_y+1)*(N/2);
				
				field_odd[i_O] += 0.25*omega*( field_even[L] + field_even[R] + field_even[D] + field_even[U] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
			}
		}
	}
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
