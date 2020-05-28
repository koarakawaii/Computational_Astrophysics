#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double EVALUATE_ERROR(int, double, double*, double*, double*, double*, int**, int**);
void INITIALIZE(int, double, double*, double*, double*, double*, double*, double*);
void LAPLACIAN_SOR(int, double, double, double*, double*, double*, double*, int**, int**);
void FPRINTF(FILE*, int N, double*);

int main(void)
{
	char filename[100];
	double optimal_omega[8] = {1.4625, 1.675, 1.8175, 1.9075, 1.9525, 1.9775, 1.99, 1.9940};
	int N_threads, display_interval;
	float preparation_time, computation_time, total_time;
	double omega, dx, criteria;
	long iter, iter_max;
	int **neighbor_even, **neighbor_odd;
	FILE* output_N_lattice;
	printf("Solve the Poission problem using SOR by OpenMP.\n\n");
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
	printf("The number of threads is %d .\n", N_threads);
	printf("\n");

	sprintf(filename, "N_lattice_OpenMP_N_threads=%d_allocate.txt",N_threads);
	output_N_lattice = fopen(filename,"w");
	fprintf(output_N_lattice, "#N_lattice\tIterations\tError\tPreparation_Time(s)\tCalculation_Time(s)\n");
	
	int count = 0;
	for (int N=8; N<=1024; N*=2)
	{
		double start = omp_get_wtime();	
		double *field_even, *field_odd, *rho_even, *rho_odd, *field_final, *field_analytic, *rho;
		int **neighbor_even, **neighbor_odd;

		omega = optimal_omega[count];
		printf("Start Preparation...\n");
		dx = 1./(N-1);
		field_even = calloc((N*N)/2,sizeof(double));
		field_odd = calloc((N*N)/2,sizeof(double));
		field_final = malloc(N*N*sizeof(double));
		field_analytic = malloc(N*N*sizeof(double));
		rho_even = malloc((N*N)/2*sizeof(double));
		rho_odd = malloc((N*N)/2*sizeof(double));
		rho = malloc(N*N*sizeof(double));
		neighbor_even = malloc(4*sizeof(int*));
		neighbor_odd = malloc(4*sizeof(int*));
	
		// construct neighbor index
		for (int i=0; i<4; i++)
		{
			neighbor_even[i] = calloc((N*N)/2,sizeof(int));
			neighbor_odd[i] = calloc((N*N)/2,sizeof(int));
		}
		//
	
		omp_set_num_threads(N_threads);

#		pragma omp parallel for
			for (int i_E=0; i_E<(N*N)/2; i_E++)
			{
				// construc neighbors for eahc even site
				int ix = i_E%(N/2);
				int iy = i_E/(N/2);
				if (iy%2==0)
				{
					if (ix!=0)
					{
						int L = ix-1 + iy*(N/2);
						int R = i_E;	
						int D = ix + (iy-1)*(N/2);
						int U = ix + (iy+1)*(N/2);
	
						neighbor_even[0][i_E] = L;
						neighbor_even[1][i_E] = R;
						neighbor_even[2][i_E] = D;
						neighbor_even[3][i_E] = U;
						
					}
				}		
				else
				{
					if (ix!=(N/2)-1)
					{
						int L = i_E;
						int R = ix+1 + iy*(N/2);	
						int D = ix + (iy-1)*(N/2);
						int U = ix + (iy+1)*(N/2);
						
						neighbor_even[0][i_E] = L;
						neighbor_even[1][i_E] = R;
						neighbor_even[2][i_E] = D;
						neighbor_even[3][i_E] = U;
					}
				}
				//
			}
#		pragma omp parallel for
			for (int i_O=0; i_O<(N*N)/2; i_O++)
			{
				// construct neighbors for each odd site
				int ix = i_O%(N/2);
				int iy = i_O/(N/2);
				if (iy%2==0)
				{
					if (ix!=(N/2)-1)
					{
						int L = i_O;	
						int R = ix+1 + iy*(N/2);
						int D = ix + (iy-1)*(N/2);
						int U = ix + (iy+1)*(N/2);
						
						neighbor_odd[0][i_O] = L;
						neighbor_odd[1][i_O] = R;
						neighbor_odd[2][i_O] = D;
						neighbor_odd[3][i_O] = U;
					}
				}		
				else
				{
					if (ix!=0)
					{
						int L = ix-1 + iy*(N/2);	
						int R = i_O;
						int D = ix + (iy-1)*(N/2);
						int U = ix + (iy+1)*(N/2);
						
						neighbor_odd[0][i_O] = L;
						neighbor_odd[1][i_O] = R;
						neighbor_odd[2][i_O] = D;
						neighbor_odd[3][i_O] = U;
					}
				}
				//
			}

		field_even = calloc((N*N)/2,sizeof(double));
		field_odd = calloc((N*N)/2,sizeof(double));
		INITIALIZE(N, dx, field_even, field_odd, field_analytic, rho_even, rho_odd, rho);
		double end = omp_get_wtime();	
		printf("Preparation ends.\n");
		preparation_time = end-start;
		printf("Total preparation time is %.4f s.\n", preparation_time);
	
		start = omp_get_wtime();	
		double error = EVALUATE_ERROR(N, dx, field_even, field_odd, rho_even, rho_odd, neighbor_even, neighbor_odd);
		printf("Starts computation with N = %d , error = %.8e ...\n", N, error);
		iter = 0;
		while (error>criteria&&iter<iter_max)
		{
			LAPLACIAN_SOR(N, dx, omega, field_even, field_odd, rho_even, rho_odd, neighbor_even, neighbor_odd);
			error = EVALUATE_ERROR(N, dx, field_even, field_odd, rho_even, rho_odd, neighbor_even, neighbor_odd);
			iter += 1;
			if (iter%display_interval==0)
				printf("Iteration = %ld , error = %.8e .\n", iter, error);
		}
	
#		pragma omp parallel for 
			for (int i_E=0; i_E<(N*N)/2; i_E++)
			{
				int ix = (2*i_E)%N;		
				int iy = (2*i_E)/N;
				int parity = (ix+iy)%2;
				ix += parity;
				field_final[ix+iy*N] = field_even[i_E];
			}
#		pragma omp parallel for 
			for (int i_O=0; i_O<(N*N)/2; i_O++)
			{
				int ix = (2*i_O)%N;		
				int iy = (2*i_O)/N;
				int parity = (ix+iy+1)%2;
				ix += parity;
				field_final[ix+iy*N] = field_odd[i_O];
			}
	
		end = omp_get_wtime();	
		computation_time = end-start;
		printf("Computation time is %.4f s.\n", computation_time);
		total_time = preparation_time + computation_time;
		printf("Total iteration is %ld ; total time is %.4f s.\n\n", iter, total_time);
		fprintf(output_N_lattice, "%d\t%ld\t%.8e\t%.4f\t%.4f\n", N, iter, error, preparation_time, computation_time);
		count += 1;
		
		free(field_even);
		free(field_odd);
		free(rho_even);
		free(rho_odd);
		free(rho);
		free(field_analytic);
		free(field_final);
		free(neighbor_even);
		free(neighbor_odd);
	}

	fclose(output_N_lattice);
	return EXIT_SUCCESS;
}

double EVALUATE_ERROR(int N, double dx, double* field_even, double* field_odd, double* rho_even, double* rho_odd, int **neighbor_even, int **neighbor_odd)
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
				error += pow(( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - 4.*field_even[i_E])/(dx*dx)-rho_even[i_E], 2.);
		}		
		else
		{
			if (i_x!=(N/2)-1)
				error += pow(( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - 4.*field_even[i_E])/(dx*dx)-rho_even[i_E], 2.);
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
				error += pow(( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - 4.*field_odd[i_O])/(dx*dx)-rho_odd[i_O], 2.);
		}		
		else
		{
			if (i_x!=0)
				error += pow(( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - 4.*field_odd[i_O])/(dx*dx)-rho_odd[i_O], 2.);
		}
	}
	return sqrt(error/(double)((N-2)*(N-2)));
}

void INITIALIZE(int N, double dx, double* field_even, double* field_odd, double* field_analytic, double* rho_even, double* rho_odd, double* rho)
{
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
}

void LAPLACIAN_SOR(int N, double dx, double omega, double* field_even, double* field_odd, double *rho_even, double *rho_odd, int **neighbor_even, int **neighbor_odd)
{
#	pragma omp parallel for
	for (int i_E=N/2; i_E<(N*N)/2-N/2; i_E++)
	{
		int i_x = i_E%(N/2);
		int i_y = i_E/(N/2);
		if (i_y%2==0)
		{
			if (i_x!=0)
				field_even[i_E] += 0.25*omega*( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
		}		
		else
		{
			if (i_x!=(N/2)-1)
				field_even[i_E] += 0.25*omega*( field_odd[neighbor_even[0][i_E]] + field_odd[neighbor_even[1][i_E]] + field_odd[neighbor_even[2][i_E]] + field_odd[neighbor_even[3][i_E]] - dx*dx*rho_even[i_E] - 4.*field_even[i_E]);
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
				field_odd[i_O] += 0.25*omega*( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
		}		
		else
		{
			if (i_x!=0)
				field_odd[i_O] += 0.25*omega*( field_even[neighbor_odd[0][i_O]] + field_even[neighbor_odd[1][i_O]] + field_even[neighbor_odd[2][i_O]] + field_even[neighbor_odd[3][i_O]] - dx*dx*rho_odd[i_O] - 4.*field_odd[i_O]);
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
