#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <cblas.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

void FPRINTF(FILE*, int, double, double*);
void COPY_NEIGHBOR(int, int, int, double*, double*, double*);
void INITIALIZE(int, int, int, int, const int, double, double*, double*, double*, double*, FILE*, FILE*);
void EVALUATE_ERROR(int, int, int, int, const int, double, double*, double*, double*, double*, double*, double*);
void LAPLACIAN(int, int, int, int, double, double, double*, double*, double*, double*);
void DAXPY(int, double, double*, double*);

int main(int argc, char* argv[])
{
	int my_rank, N_rank;
	const int root_rank = 0;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &N_rank);

	int N, N_per_rank, shift, display_interval;
	float preparation_time, computation_time, total_time;
	double photon_mass, dx, criteria, error, error_rank, norm, norm_rank;
	double alpha, beta;
	double start, end;
	long iter, iter_max;
	double *field, *field_rank, *rho, *rho_rank, *neighbor_U, *neighbor_D, *r, *p, *A_p, *field_analytic;
	size_t size_lattice, size_lattice_rank;
	FILE* output_field, *output_rho;

	if (my_rank==root_rank)
	{
		printf("Solve the Poission problem using CG by MPI.\n\n");
		printf("Enter the latttice size (N,N) (N must be divisible by 2).");
		scanf("%d", &N);
		if (N%N_rank!=0)
		{
			printf("N is not divisible by number of processors! Exit!\n");
			return EXIT_FAILURE;
		}
		else
		{
			printf("The lattice size is (%d,%d).\n", N, N);
			N_per_rank = N/N_rank;
		}
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
		printf("\n");
		printf("Start Preparation...\n");
		start = omp_get_wtime();
	}

	MPI_Bcast( &N, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
	MPI_Bcast( &photon_mass, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Bcast( &iter_max, 1, MPI_LONG, root_rank, MPI_COMM_WORLD);
	MPI_Bcast( &criteria, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Bcast( &display_interval, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

	dx = 1./(N-1);	
	N_per_rank = N/N_rank;
	shift = N_per_rank*N;
	size_lattice_rank = shift*sizeof(double);
	field_rank = malloc(size_lattice_rank);
	rho_rank = malloc(size_lattice_rank);
	r = malloc(size_lattice_rank);
	p = malloc(size_lattice_rank);
	A_p = malloc(size_lattice_rank);

	if (N_rank>1)
    {
        if (my_rank==0)
        {
            neighbor_U = malloc(N*sizeof(double));
            neighbor_D = NULL;
        }
        else if (my_rank==N_rank-1)
        {
            neighbor_D = malloc(N*sizeof(double));
            neighbor_U = NULL;
        }
        else
        {
            neighbor_U = malloc(N*sizeof(double));
            neighbor_D = malloc(N*sizeof(double));
        }
    }
    else
    {
        neighbor_U = NULL;
        neighbor_D = NULL;
    }

	if (my_rank==0)
	{
		size_lattice = N*N*sizeof(double);
		field = malloc(size_lattice);
		field_analytic = malloc(size_lattice);
		rho = malloc(size_lattice);
		output_field = fopen("analytical_field_distribution_CG_MPI.txt","w");
		output_rho = fopen("charge_distribution_CG.txt","w");
	}

	INITIALIZE(N_per_rank, N, shift, my_rank, root_rank, dx, rho, rho_rank, field_analytic, field_rank, output_field, output_rho);
	norm_rank = cblas_ddot(shift, rho_rank, 1, rho_rank, 1);
	MPI_Reduce(&norm_rank, &norm, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);

	if (my_rank == root_rank)
	{
		norm = sqrt(norm);
//		printf("Norm = %.4e .\n",norm);
	}
	MPI_Bcast( &norm, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);

	COPY_NEIGHBOR(N, shift, my_rank, field_rank, neighbor_U, neighbor_D);
	EVALUATE_ERROR(N_per_rank, N, shift, my_rank, root_rank, norm, rho_rank, field_rank, neighbor_U, neighbor_D, &error_rank, &error);
	memcpy(r, rho_rank, size_lattice_rank);
	memcpy(p, rho_rank, size_lattice_rank);
	
	if (my_rank==0)
	{
		double end = omp_get_wtime();
		printf("Preparation ends.\n");
		preparation_time = end - start;
		printf("Total preparation time is %.4f s.\n\n", preparation_time);
		printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
		start = omp_get_wtime();
	}

	double temp_rank, temp;
	iter = 0;

	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		COPY_NEIGHBOR(N, shift, my_rank, p, neighbor_U, neighbor_D);
		MPI_Barrier(MPI_COMM_WORLD);
		LAPLACIAN(N_per_rank, N, shift, my_rank, dx, photon_mass, p, A_p, neighbor_U, neighbor_D);
		temp_rank = cblas_ddot(shift, p, 1, A_p, 1);
//		printf("%d\t%.4f\n", my_rank, temp_rank);
		MPI_Reduce(&temp_rank, &temp, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
		MPI_Bcast( &temp, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
		alpha = error/temp;
		cblas_daxpy(shift, -alpha, A_p, 1, r, 1);
		cblas_daxpy(shift, alpha, p, 1, field_rank, 1);
		temp_rank = cblas_ddot(shift, r, 1, r, 1);
		MPI_Reduce(&temp_rank, &temp, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
		MPI_Bcast( &temp, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
		beta = temp/error;
//		printf("%.4f\t%.4f\n", alpha, beta);
		DAXPY(shift, beta, p, r);
		error = temp;
		iter += 1;
		if (iter%display_interval==0&&my_rank==root_rank)
			printf("Iteration = %ld , error = %.8e .\n", iter, sqrt(error)/norm);
	}
  
	MPI_Gather(field_rank, shift, MPI_DOUBLE, field, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	if (my_rank==root_rank)
	{
		output_field = fopen("simulated_field_distribution_MPI_CG.txt","w");
		FPRINTF(output_field, N, 1., field);
		end = omp_get_wtime();
		computation_time = end - start;
		printf("Computation time is %.4f s.\n", computation_time);
		total_time = preparation_time + computation_time;
		printf("Total iteration is %ld ; total time is %.4f s.\n", iter, total_time);

		free(field);
		fclose(output_field);
	}

	free(r);
	free(p);
	free(A_p);
	MPI_Finalize();
	return EXIT_SUCCESS;
}

void COPY_NEIGHBOR(int N, int shift, int my_rank, double* src, double* neighbor_U, double* neighbor_D)
{
    const int tag_0 = 0, tag_1 = 1;
    MPI_Request req_1[2];
    MPI_Request req_2[2];
    if (neighbor_D!=NULL)
    {
        MPI_Irecv(neighbor_D, N, MPI_DOUBLE, my_rank-1, tag_1, MPI_COMM_WORLD, &req_1[0]);
        MPI_Isend(src, N, MPI_DOUBLE, my_rank-1, tag_0, MPI_COMM_WORLD, &req_1[1]);
        MPI_Waitall(2,req_1, MPI_STATUSES_IGNORE);
    }
    if (neighbor_U!=NULL)
    {
        MPI_Irecv(neighbor_U, N, MPI_DOUBLE, my_rank+1, tag_0, MPI_COMM_WORLD, &req_2[0]);
        MPI_Isend(src+shift-N, N, MPI_DOUBLE, my_rank+1, tag_1, MPI_COMM_WORLD, &req_2[1]);
        MPI_Waitall(2,req_2, MPI_STATUSES_IGNORE);
    }
}

void INITIALIZE(int N_per_rank, int N, int shift, int my_rank, const int root_rank, double dx, double* rho, double* rho_rank, double* field_analytic, double* field_rank, FILE* output_field, FILE* output_rho)
{
	int idx_x, idx_y, idx_y_lattice;
	double x, y;
	double* temp = malloc(shift*sizeof(double));

	for (int idx=0; idx<shift; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		idx_y_lattice = idx_y + N_per_rank*my_rank;
	
		x = idx_x*dx;
		y = (idx_y_lattice)*dx;
		temp[idx] = x*(1.-x)*y*(1.-y)*exp(x-y);
	
		if (idx_x!=0&&idx_x!=N-1&&idx_y_lattice!=0&&idx_y_lattice!=N-1)
		{
			field_rank[idx] = 0.0;
			rho_rank[idx] = (2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y))*dx*dx;	// Notice that rho has been times by dx^2!!
		}
		else
		{
			field_rank[idx] = temp[idx];
			rho_rank[idx] = 0.0;
		}
	}

	MPI_Gather(temp, shift, MPI_DOUBLE, field_analytic, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Gather(rho_rank, shift, MPI_DOUBLE, rho, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	
	if (my_rank==root_rank)
	{
		FPRINTF(output_field, N, 1., field_analytic);
		FPRINTF(output_rho, N, pow(dx,-2.), rho);
		fclose(output_rho);
		free(rho);
		free(field_analytic);
	}
	free(temp);
}

void EVALUATE_ERROR(int N_per_rank, int N, int shift, int my_rank, const int root_rank, double norm, double* rho_rank, double* field_rank, double* neighbor_U, double* neighbor_D, double* error_rank, double* error)
{
	int idx_x, idx_y;
	double L, R, U, D;
	*error_rank = 0.0;

	for (int idx=0; idx<shift; idx++)
	{
		idx_x = idx%N;
		idx_y = (idx+my_rank*shift)/N;
	
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			L = field_rank[idx-1];
			R = field_rank[idx+1];
			if (idx/N!=N_per_rank-1)	
				U = field_rank[idx+N];
			else
				U = neighbor_U[idx_x];
			if (idx/N!=0)
				D = field_rank[idx-N];
			else
				D = neighbor_D[idx_x];
;			*error_rank += pow(L + R + U + D - 4.*field_rank[idx] - rho_rank[idx], 2.);
		}
	}
	MPI_Reduce(error_rank, error, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
	MPI_Bcast(error, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
}

void LAPLACIAN(int N_per_rank, int N, int shift, int my_rank, double dx, double photon_mass, double* p, double* A_p, double *neighbor_U, double* neighbor_D)
{
	int idx_x, idx_y;
	double L, R, U, D;
	
	for (int idx=0; idx<shift; idx++)
	{
		idx_x = idx%N;
		idx_y = (idx+my_rank*shift)/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			L = p[idx-1];
			R = p[idx+1];
			if (idx/N!=N_per_rank-1)
				U = p[idx+N];
			else
				U = neighbor_U[idx_x];
			if (idx/N!=0)
				D = p[idx-N];
			else
				D = neighbor_D[idx_x];
	
			A_p[idx] = ( L + R + U + D - (4.+pow(photon_mass*dx,2.))*p[idx]);
	//		printf("%d\t%.4f\n", idx, A_p[idx]);
		}
		else
			A_p[idx] = 0.0;
	}
}

void DAXPY(int shift, double c, double *A, double *B)
{
	for (int idx=0; idx<shift; idx++)
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
