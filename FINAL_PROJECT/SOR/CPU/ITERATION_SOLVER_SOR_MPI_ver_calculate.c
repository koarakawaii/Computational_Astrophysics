/* This version use neighbor_e and neighbor_o to stroe the negibhor indices of even sites and odd site.*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <cblas.h>
#include <string.h>

void EVALUATE_ERROR(int, int, int, int, int, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*);
void COPY_NEIGHBOR_EVEN(int, int, int, double*, double*, double*);
void COPY_NEIGHBOR_ODD(int, int, int, double*, double*, double*);
void LAPLACIAN_SOR_EVEN(int, int, int, int, double, double, double*, double*, double*, double*, double*);
void LAPLACIAN_SOR_ODD(int, int, int, int, double, double, double*, double*, double*, double*, double*);
void FPRINTF(FILE*, int N, double*);

int main(int argc, char *argv[])
{
	int my_rank, N_rank, N, N_threads, display_interval;
	const int root_rank = 0;
	float preparation_time, computation_time, total_time;
	double dx, error_rank, error, criteria, omega, norm_rank, norm;
	double start, end;
	long iter_max;
	double *neighbor_U_even, *neighbor_D_even, *neighbor_U_odd, *neighbor_D_odd;
	double *field_even, *field_odd, *rho_even, *rho_odd;
	double *field_even_rank, *field_odd_rank, *rho_even_rank, *rho_odd_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &N_rank);

	if (my_rank==root_rank)
	{
		printf("Solve the Poission problem using SOR by MPI.\n");
		printf("Enter the latttice size (N,N) (N must be divisible by N_rank).\n");
		scanf("%d", &N);
		if (N%(N_rank)!=0)
		{
			printf("N must be divisible by 2*N_rank! Exit!\n");
			return EXIT_FAILURE;
		}
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
		
		start = omp_get_wtime();	
		dx = 1./(N-1);
		double* field_analytic = (double*)malloc(N*N*sizeof(double));
		double* rho = (double*)malloc(N*N*sizeof(double));
		field_even = (double*)calloc((N/2)*N,sizeof(double));
		field_odd = (double*)calloc((N/2)*N,sizeof(double));
		rho_even = (double *)calloc((N/2)*N,sizeof(double));
		rho_odd = (double *)calloc((N/2)*N,sizeof(double));
		FILE* output_field = fopen("analytical_field_distribution.txt","w");
		FILE* output_rho = fopen("charge_distribution.txt","w");

		omp_set_num_threads(N_threads);
	#	pragma omp parallel for
		for (int i=0; i<(N*N)/2; i++)
		{
			int ix = (2*i)%N;		
			int iy = (2*i)/N;
			int parity = (ix+iy)%2;
			ix += parity;
			double x = ix*dx;
			double y = iy*dx;
			field_analytic[ix+iy*N] = x*(1.-x)*y*(1.-y)*exp(x-y);
			if ( ix==0 || ix==N-1 || iy==0 || iy==N-1 )
				field_even[i] = field_analytic[ix+iy*N];
			else
			{
				rho_even[i] = 2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y);
				rho[ix+iy*N] = rho_even[i];
			}


			ix = (2*i)%N;		
			iy = (2*i)/N;
			parity = (ix+iy+1)%2;
			ix += parity;
			x = ix*dx;
			y = iy*dx;
			field_analytic[ix+iy*N] = x*(1.-x)*y*(1.-y)*exp(x-y);
			if ( ix==0 || ix==N-1 || iy==0 || iy==N-1 )
				field_odd[i] = field_analytic[ix+iy*N];
			else
			{
				rho_odd[i] = 2.*x*(y-1)*(y-2.*x+x*y+2)*exp(x-y);
				rho[ix+iy*N] = rho_odd[i];
			}
		}

		FPRINTF(output_field, N, field_analytic);
		FPRINTF(output_rho, N, rho);
		fclose(output_field);
		fclose(output_rho);
		free(field_analytic);
		free(rho);
//		end = omp_get_wtime();
//		printf("Time for initialization of charge and field distribution takes %.4f ms.\n", end-start);
	}

	MPI_Bcast( &N, 1, MPI_INT, root_rank, MPI_COMM_WORLD );
	MPI_Bcast( &N_threads, 1, MPI_INT, root_rank, MPI_COMM_WORLD );
	MPI_Bcast( &display_interval, 1, MPI_INT, root_rank, MPI_COMM_WORLD );
	MPI_Bcast( &iter_max, 1, MPI_LONG, root_rank, MPI_COMM_WORLD );
	MPI_Bcast( &omega, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD );
	MPI_Bcast( &criteria, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD );

	if (my_rank!=root_rank)
	{
		dx = 1./(N-1);
//		omp_set_num_threads(N_threads);
	}
	int shift = (N/2)*(N/N_rank);
	field_even_rank = (double*)malloc(shift*sizeof(double));
	field_odd_rank = (double*)malloc(shift*sizeof(double));
	rho_even_rank = (double*)malloc(shift*sizeof(double));
	rho_odd_rank = (double*)malloc(shift*sizeof(double));
	MPI_Scatter(field_even, shift, MPI_DOUBLE, field_even_rank, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Scatter(field_odd, shift, MPI_DOUBLE, field_odd_rank, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Scatter(rho_even, shift, MPI_DOUBLE, rho_even_rank, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Scatter(rho_odd, shift, MPI_DOUBLE, rho_odd_rank, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);

	norm_rank = cblas_ddot(shift, rho_even_rank, 1, rho_even_rank, 1);
	norm_rank += cblas_ddot(shift, rho_odd_rank, 1, rho_odd_rank, 1);
    MPI_Reduce(&norm_rank, &norm, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);

    if (my_rank == root_rank)
    {
        norm = sqrt(norm);
//      printf("Norm = %.4e .\n",norm);
    }
    MPI_Bcast( &norm, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);	
	
	if (N_rank>1)
	{
		if (my_rank==0)
		{
			neighbor_U_even = (double*)malloc(N/2*sizeof(double));
			neighbor_D_even = NULL;
			neighbor_U_odd = (double *)malloc(N/2*sizeof(double));
			neighbor_D_odd = NULL;
		}
		else if (my_rank==N_rank-1)
		{
			neighbor_D_even = (double *)malloc(N/2*sizeof(double));
			neighbor_U_even = NULL;
			neighbor_D_odd = (double *)malloc(N/2*sizeof(double));
			neighbor_U_odd = NULL;
		}
		else
		{
			neighbor_U_even = (double *)malloc(N/2*sizeof(double));
			neighbor_D_even = (double *)malloc(N/2*sizeof(double));
			neighbor_U_odd = (double *)malloc(N/2*sizeof(double));
			neighbor_D_odd = (double *)malloc(N/2*sizeof(double));
		}
	}
	else
	{
		neighbor_U_even = NULL; 
		neighbor_D_even = NULL;
		neighbor_U_odd = NULL;
		neighbor_D_odd = NULL;
	}

//	if (my_rank==root_rank)
//		printf("Start copy...\n");
//	MPI_Barrier(MPI_COMM_WORLD);
	COPY_NEIGHBOR_EVEN(N, shift, my_rank, field_even_rank, neighbor_U_even, neighbor_D_even);
	COPY_NEIGHBOR_ODD(N, shift, my_rank, field_odd_rank, neighbor_U_odd, neighbor_D_odd);

	EVALUATE_ERROR(N, shift, my_rank, N_rank, root_rank, dx, &error_rank, &error, field_even_rank, field_odd_rank, rho_even_rank, rho_odd_rank, neighbor_U_even, neighbor_D_even, neighbor_U_odd, neighbor_D_odd);
	
	if (my_rank==root_rank)
	{
		end = omp_get_wtime();
		preparation_time = end-start;
		printf("Total preparation time is %.4f s.\n\n", preparation_time);
		printf("Starts computation with error = %.8e...\n", sqrt(error)/norm);
		start = omp_get_wtime();
	}

//	MPI_Barrier(MPI_COMM_WORLD);
	int iter = 0;
	while (sqrt(error)/norm>criteria&&iter<iter_max)
	{
		LAPLACIAN_SOR_EVEN(N, shift, my_rank, N_rank, dx, omega, field_even_rank, field_odd_rank, rho_even_rank, neighbor_U_odd, neighbor_D_odd);
//		MPI_Barrier(MPI_COMM_WORLD);
		COPY_NEIGHBOR_EVEN(N, shift, my_rank, field_even_rank, neighbor_U_even, neighbor_D_even);

		LAPLACIAN_SOR_ODD(N, shift, my_rank, N_rank, dx, omega, field_even_rank, field_odd_rank, rho_odd_rank, neighbor_U_even, neighbor_D_even);
//		MPI_Barrier(MPI_COMM_WORLD);
		COPY_NEIGHBOR_ODD(N, shift, my_rank, field_odd_rank, neighbor_U_odd, neighbor_D_odd);

		EVALUATE_ERROR(N, shift, my_rank, N_rank, root_rank, dx, &error_rank, &error, field_even_rank, field_odd_rank, rho_even_rank, rho_odd_rank, neighbor_U_even, neighbor_D_even, neighbor_U_odd, neighbor_D_odd);

		iter += 1;
		if (iter%display_interval==0&&my_rank==root_rank)
			printf("Iteration = %d , error = %.8e .\n", iter, sqrt(error)/norm);
	}

	MPI_Gather(field_even_rank, shift, MPI_DOUBLE, field_even, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
	MPI_Gather(field_odd_rank, shift, MPI_DOUBLE, field_odd, shift, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);

	if (my_rank==root_rank)
	{
		double* field_final = (double*)malloc(N*N*sizeof(double));
#	pragma omp parallel for 
		for (int i=0; i<(N*N)/2; i++)
		{
			int ix = (2*i)%N;		
			int iy = (2*i)/N;
			int parity = (ix+iy)%2;
			ix += parity;
			field_final[ix+iy*N] = field_even[i];

			ix = (2*i)%N;		
			iy = (2*i)/N;
			parity = (ix+iy+1)%2;
			ix += parity;
			field_final[ix+iy*N] = field_odd[i];
		}
  	
		FILE* output_field = fopen("simulated_field_distribution_MPI_calculate.txt","w");
		FPRINTF(output_field, N, field_final);
		end = omp_get_wtime();
		computation_time = end-start;
		printf("Computation time is %.4f s.\n", computation_time);
		total_time = preparation_time + computation_time;
		printf("Total iteration is %d ; total time is %.4f s.\n", iter, total_time);

		free(field_final);
		free(field_even);
		free(field_odd);
		free(rho_odd);
		free(rho_even);
		fclose(output_field);
	}

	free(field_even_rank);
	free(field_odd_rank);
	free(rho_odd_rank);
	free(rho_even_rank);

	MPI_Finalize();
	return EXIT_SUCCESS;
}

void COPY_NEIGHBOR_EVEN(int N, int shift, int my_rank, double* field_even_rank, double* neighbor_U_even, double* neighbor_D_even)
{
	const int tag_0 = 0, tag_1 = 1;
	MPI_Request req_1[2];
	MPI_Request req_2[2];
	if (neighbor_D_even!=NULL)
	{
		MPI_Irecv(neighbor_D_even, N/2, MPI_DOUBLE, my_rank-1, tag_1, MPI_COMM_WORLD, &req_1[0]);		
		MPI_Isend(field_even_rank, N/2, MPI_DOUBLE, my_rank-1, tag_0, MPI_COMM_WORLD, &req_1[1]);
		MPI_Waitall(2,req_1, MPI_STATUSES_IGNORE);
	}
	if (neighbor_U_even!=NULL)
	{
		MPI_Irecv(neighbor_U_even, N/2, MPI_DOUBLE, my_rank+1, tag_0, MPI_COMM_WORLD, &req_2[0]);		
		MPI_Isend(field_even_rank+shift-N/2, N/2, MPI_DOUBLE, my_rank+1, tag_1, MPI_COMM_WORLD, &req_2[1]);
		MPI_Waitall(2,req_2, MPI_STATUSES_IGNORE);
	}
}

void COPY_NEIGHBOR_ODD(int N, int shift, int my_rank, double* field_odd_rank, double* neighbor_U_odd, double* neighbor_D_odd)
{
	const int tag_0 = 0, tag_1 = 1;
	MPI_Request req_1[2];
	MPI_Request req_2[2];
	if (neighbor_D_odd!=NULL)
	{
		MPI_Irecv(neighbor_D_odd, N/2, MPI_DOUBLE, my_rank-1, tag_1, MPI_COMM_WORLD, &req_1[0]);		
		MPI_Isend(field_odd_rank, N/2, MPI_DOUBLE, my_rank-1, tag_0, MPI_COMM_WORLD, &req_1[1]);
		MPI_Waitall(2,req_1, MPI_STATUSES_IGNORE);
	}
	if (neighbor_U_odd!=NULL)
	{
		MPI_Irecv(neighbor_U_odd, N/2, MPI_DOUBLE, my_rank+1, tag_0, MPI_COMM_WORLD, &req_2[0]);		
		MPI_Isend(field_odd_rank+shift-N/2, N/2, MPI_DOUBLE, my_rank+1, tag_1, MPI_COMM_WORLD, &req_2[1]);
		MPI_Waitall(2,req_2, MPI_STATUSES_IGNORE);
	}
}

void EVALUATE_ERROR(int N, int shift, int my_rank, int N_rank, int root_rank, double dx, double* E_block, double* E, double* field_even_rank, double* field_odd_rank, double* rho_even_rank, double* rho_odd_rank, double* neighbor_U_even, double* neighbor_D_even, double* neighbor_U_odd, double* neighbor_D_odd)
{
	*E = 0.0;
	*E_block = 0.0;
	double L, R, U, D;
	if (shift!=N/2)
	{
		for (int i=N/2; i<shift-N/2; i++)
		{
			int i_x = (i+shift*my_rank)%(N/2);
			int i_y = (i+shift*my_rank)/(N/2);
			if (i_y%2==0)
			{
				if (i_x!=0)
				{
					L = field_odd_rank[i-1];
					R = field_odd_rank[i];
					U = field_odd_rank[i+N/2];
					D = field_odd_rank[i-N/2];
					*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
//					printf("%d\t%.6f\n", my_rank, pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i],2.) );
				}
				if (i_x!=(N/2)-1)
				{
					L = field_even_rank[i];
					R = field_even_rank[i+1];
					U = field_even_rank[i+N/2];
					D = field_even_rank[i-N/2];
					*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
//					printf("%d\t%.6f\n", my_rank, pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i],2.) );
				}
			}		
			else
			{
				if (i_x!=(N/2)-1)
				{
					L = field_odd_rank[i];
					R = field_odd_rank[i+1];
					U = field_odd_rank[i+N/2];
					D = field_odd_rank[i-N/2];
					*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
//					printf("%d\t%.6f\n", my_rank, pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i],2.) );
				}
				if (i_x!=0)
				{
					L = field_even_rank[i-1];
					R = field_even_rank[i];
					U = field_even_rank[i+N/2];
					D = field_even_rank[i-N/2];
					*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
//					printf("%d\t%.6f\n", my_rank, pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i],2.) );
				}
			}
		}
		if (my_rank==0)
		{
			if (neighbor_U_even!=NULL)
			{
				for (int i=shift-N/2; i<shift; i++)
				{
					int i_x = i%(N/2);
					int i_y = i/(N/2);
		
					if (i_y%2==0)
					{
						if (i_x!=0)
						{
							L = field_odd_rank[i-1];
							R = field_odd_rank[i];
							U = neighbor_U_odd[i_x];
							D = field_odd_rank[i-N/2];
							*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
						}
						if (i_x!=(N/2)-1)	
						{
							L = field_even_rank[i];
							R = field_even_rank[i+1];
							U = neighbor_U_even[i_x];
							D = field_even_rank[i-N/2];
							*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
						}
					}
					else
					{
						if (i_x!=(N/2)-1)
						{
							L = field_odd_rank[i];
							R = field_odd_rank[i+1];
							U = neighbor_U_odd[i_x];
							D = field_odd_rank[i-N/2];
							*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
						}
						if (i_x!=0)
						{
							L = field_even_rank[i-1];
							R = field_even_rank[i];
							U = neighbor_U_even[i_x];
							D = field_even_rank[i-N/2];
							*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
						}
					}
				}
			}
		}
		else if (my_rank==N_rank-1)
		{
			if (neighbor_D_even!=NULL)
			{
				for (int i=0; i<N/2; i++)
				{
					int i_x = (i+(N_rank-1)*shift)%(N/2);
					int i_y = (i+(N_rank-1)*shift)/(N/2);
		
					if (i_y%2==0)
					{
						if (i_x!=0)
						{
							L = field_odd_rank[i-1];
							R = field_odd_rank[i];
							U = field_odd_rank[i+N/2];
							D = neighbor_D_odd[i_x];
							*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
						}
						if (i_x!=(N/2)-1)	
						{
							L = field_even_rank[i];
							R = field_even_rank[i+1];
							U = field_even_rank[i+N/2];
							D = neighbor_D_even[i_x];
							*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
						}
					}
					else
					{
						if (i_x!=(N/2)-1)
						{
							L = field_odd_rank[i];
							R = field_odd_rank[i+1];
							U = field_odd_rank[i+N/2];
							D = neighbor_D_odd[i_x];
							*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
						}
						if (i_x!=0)
						{
							L = field_even_rank[i-1];
							R = field_even_rank[i];
							U = field_even_rank[i+N/2];
							D = neighbor_D_even[i_x];
							*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
						}
					}
				}
			}
		}
		else
		{
			for (int i=0; i<N/2; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
	
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = field_odd_rank[i+N/2];
						D = neighbor_D_odd[i_x];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=(N/2)-1)
					{	
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = field_even_rank[i+N/2];
						D = neighbor_D_even[i_x];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = field_odd_rank[i+N/2];
						D = neighbor_D_odd[i_x];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = field_even_rank[i+N/2];
						D = neighbor_D_even[i_x];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
			}
			for (int i=shift-N/2; i<shift; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
	
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = neighbor_U_odd[i_x];
						D = field_odd_rank[i-N/2];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=(N/2)-1)
					{	
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = neighbor_U_even[i_x];
						D = field_even_rank[i-N/2];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = neighbor_U_odd[i_x];
						D = field_odd_rank[i-N/2];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = neighbor_U_even[i_x];
						D = field_even_rank[i-N/2];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
			}
		}
	} // shift!=N/2
	else
	{
		if (my_rank!=0 && my_rank!=N_rank-1)
		{
			for (int i=0; i<N/2;i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
	
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = neighbor_U_odd[i_x];
						D = neighbor_D_odd[i_x];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=(N/2)-1)	
					{
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = neighbor_U_even[i_x];
						D = neighbor_D_even[i_x];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = neighbor_U_odd[i_x];
						D = neighbor_D_odd[i_x];
						*E_block += pow((L+R+U+D-4.*field_even_rank[i])/dx/dx-rho_even_rank[i], 2.);
					}
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = neighbor_U_even[i_x];
						D = neighbor_D_even[i_x];
						*E_block += pow((L+R+U+D-4.*field_odd_rank[i])/dx/dx-rho_odd_rank[i], 2.);
					}
				}
			}
		}
	}
//	printf("%d\t%.6f\n", my_rank, *E_block);
	MPI_Reduce(E_block, E, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
//	Broadcast to all the ranks
	MPI_Bcast( E, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD );
}

void LAPLACIAN_SOR_EVEN(int N, int shift, int my_rank, int N_rank, double dx, double omega, double* field_even_rank, double* field_odd_rank, double *rho_even_rank, double* neighbor_U_odd, double* neighbor_D_odd)
{
	double L, R, U, D;
	if (shift!=N/2)
	{
		for (int i=N/2; i<shift-N/2; i++)
		{
			int i_x = (i+shift*my_rank)%(N/2);
			int i_y = (i+shift*my_rank)/(N/2);
			if (i_y%2==0)
			{
				if (i_x!=0)
				{
					L = field_odd_rank[i-1];
					R = field_odd_rank[i];
					U = field_odd_rank[i+N/2];
					D = field_odd_rank[i-N/2];
					field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//					printf("%d\t%.6f\n", i, field_even_rank[i]);
//					printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
				}
			}		
			else
			{
				if (i_x!=(N/2)-1)
				{
					L = field_odd_rank[i];
					R = field_odd_rank[i+1];
					U = field_odd_rank[i+N/2];
					D = field_odd_rank[i-N/2];
					field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//					printf("%d\t%.6f\n", i, field_even_rank[i]);
//					printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
				}
			}
		}
		if (my_rank==0)
		{
			if (neighbor_U_odd!=NULL)
			{
				for (int i=shift-N/2; i<shift; i++)
				{
					int i_x = i%(N/2);
					int i_y = i/(N/2);
					if (i_y%2==0)
					{
						if (i_x!=0)
						{
							L = field_odd_rank[i-1];
							R = field_odd_rank[i];
							U = neighbor_U_odd[i_x];
							D = field_odd_rank[i-N/2];
							field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//							printf("%d\t%.6f\n", i, field_even_rank[i]);
						}
					}
					else
					{
						if (i_x!=(N/2)-1)
						{
							L = field_odd_rank[i];
							R = field_odd_rank[i+1];
							U = neighbor_U_odd[i_x];
							D = field_odd_rank[i-N/2];
							field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//							printf("%d\t%.6f\n", i, field_even_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
						}
					}
				}
			}
		}
		else if (my_rank==N_rank-1)
		{
			if (neighbor_D_odd!=NULL)
			{
				for (int i=0; i<N/2; i++)
				{
					int i_x = (i+(N_rank-1)*shift)%(N/2);
					int i_y = (i+(N_rank-1)*shift)/(N/2);
					if (i_y%2==0)
					{
						if (i_x!=0)
						{
							L = field_odd_rank[i-1];
							R = field_odd_rank[i];
							U = field_odd_rank[i+N/2];
							D = neighbor_D_odd[i_x];
							field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//							printf("%d\t%.6f\n", i, field_even_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
						}
					}
					else
					{
						if (i_x!=(N/2)-1)
						{
							L = field_odd_rank[i];
							R = field_odd_rank[i+1];
							U = field_odd_rank[i+N/2];
							D = neighbor_D_odd[i_x];
							field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//							printf("%d\t%.6f\n", i, field_even_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
						}
					}
				}
			}
		}
		else
		{
			for (int i=0; i<N/2; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = field_odd_rank[i+N/2];
						D = neighbor_D_odd[i_x];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//						printf("%d\t%.6f\n", i, field_even_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = field_odd_rank[i+N/2];
						D = neighbor_D_odd[i_x];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//						printf("%d\t%.6f\n", i, field_even_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
					}
				}
			}
			for (int i=shift-N/2; i<shift; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = neighbor_U_odd[i_x];
						D = field_odd_rank[i-N/2];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//						printf("%d\t%.6f\n", i, field_even_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = neighbor_U_odd[i_x];
						D = field_odd_rank[i-N/2];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
//						printf("%d\t%.6f\n", i, field_even_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_even_rank[i]);
					}
				}
			}
		}
	} // shift!=N/2
	else
	{
		if (my_rank!=0 && my_rank!=N_rank-1)
		{
			for (int i=0; i<N/2;i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=0)
					{
						L = field_odd_rank[i-1];
						R = field_odd_rank[i];
						U = neighbor_U_odd[i_x];
						D = neighbor_D_odd[i_x];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
					}
				}
				else
				{
					if (i_x!=(N/2)-1)
					{
						L = field_odd_rank[i];
						R = field_odd_rank[i+1];
						U = neighbor_U_odd[i_x];
						D = neighbor_D_odd[i_x];
						field_even_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_even_rank[i]-4.*field_even_rank[i]);
					}
				}
			}
		}
	}
}

void LAPLACIAN_SOR_ODD(int N, int shift, int my_rank, int N_rank, double dx, double omega, double* field_even_rank, double* field_odd_rank, double *rho_odd_rank, double* neighbor_U_even, double* neighbor_D_even)
{
	double L, R, U, D;
	if (shift!=N/2)
	{
		for (int i=N/2; i<shift-N/2; i++)
		{
			int i_x = (i+shift*my_rank)%(N/2);
			int i_y = (i+shift*my_rank)/(N/2);
			if (i_y%2==0)
			{
				if (i_x!=(N/2)-1)
				{
					L = field_even_rank[i];
					R = field_even_rank[i+1];
					U = field_even_rank[i+N/2];
					D = field_even_rank[i-N/2];
					field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//					printf("%d\t%.6f\n", i, field_odd_rank[i]);
//					printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
				}
			}		
			else
			{
				if (i_x!=0)
				{
					L = field_even_rank[i-1];
					R = field_even_rank[i];
					U = field_even_rank[i+N/2];
					D = field_even_rank[i-N/2];
					field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//					printf("%d\t%.6f\n", i, field_odd_rank[i]);
//					printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
				}
			}
		}
		if (my_rank==0)
		{
			if (neighbor_U_even!=NULL)
			{
				for (int i=shift-N/2; i<shift; i++)
				{
					int i_x = i%(N/2);
					int i_y = i/(N/2);
					if (i_y%2==0)
					{
						if (i_x!=(N/2)-1)	
						{
							L = field_even_rank[i];
							R = field_even_rank[i+1];
							U = neighbor_U_even[i_x];
							D = field_even_rank[i-N/2];
							field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//							printf("%d\t%.6f\n", i, field_odd_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
						}
					}
					else
					{
						if (i_x!=0)
						{
							L = field_even_rank[i-1];
							R = field_even_rank[i];
							U = neighbor_U_even[i_x];
							D = field_even_rank[i-N/2];
							field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//							printf("%d\t%.6f\n", i, field_odd_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
						}
					}
				}
			}
		}
		else if (my_rank==N_rank-1)
		{
			if (neighbor_D_even!=NULL)
			{
				for (int i=0; i<N/2; i++)
				{
					int i_x = (i+(N_rank-1)*shift)%(N/2);
					int i_y = (i+(N_rank-1)*shift)/(N/2);
					if (i_y%2==0)
					{
						if (i_x!=(N/2)-1)	
						{
							L = field_even_rank[i];
							R = field_even_rank[i+1];
							U = field_even_rank[i+N/2];
							D = neighbor_D_even[i_x];
							field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//							printf("%d\t%.6f\n", i, field_odd_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
						}
					}
					else
					{
						if (i_x!=0)
						{
							L = field_even_rank[i-1];
							R = field_even_rank[i];
							U = field_even_rank[i+N/2];
							D = neighbor_D_even[i_x];
							field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//							printf("%d\t%.6f\n", i, field_odd_rank[i]);
//							printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
						}
					}
				}
			}
		}
		else
		{
			for (int i=0; i<N/2; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=(N/2)-1)
					{	
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = field_even_rank[i+N/2];
						D = neighbor_D_even[i_x];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//						printf("%d\t%.6f\n", i, field_odd_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
					}
				}
				else
				{
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = field_even_rank[i+N/2];
						D = neighbor_D_even[i_x];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//						printf("%d\t%.6f\n", i, field_odd_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
					}
				}
			}
			for (int i=shift-N/2; i<shift; i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=(N/2)-1)
					{	
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = neighbor_U_even[i_x];
						D = field_even_rank[i-N/2];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//						printf("%d\t%.6f\n", i, field_odd_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
					}
				}
				else
				{
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = neighbor_U_even[i_x];
						D = field_even_rank[i-N/2];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
//						printf("%d\t%.6f\n", i, field_odd_rank[i]);
//						printf("%d\t%d\t%.6f\n", i_x, i_y, field_odd_rank[i]);
					}
				}
			}
		}
	} // shift!=N/2
	else
	{
		if (my_rank!=0 && my_rank!=N_rank-1)
		{
			for (int i=0; i<N/2;i++)
			{
				int i_x = (i+my_rank*shift)%(N/2);
				int i_y = (i+my_rank*shift)/(N/2);
				if (i_y%2==0)
				{
					if (i_x!=(N/2)-1)	
					{
						L = field_even_rank[i];
						R = field_even_rank[i+1];
						U = neighbor_U_even[i_x];
						D = neighbor_D_even[i_x];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
					}
				}
				else
				{
					if (i_x!=0)
					{
						L = field_even_rank[i-1];
						R = field_even_rank[i];
						U = neighbor_U_even[i_x];
						D = neighbor_D_even[i_x];
						field_odd_rank[i] += 0.25*omega*(L+R+D+U-dx*dx*rho_odd_rank[i]-4.*field_odd_rank[i]);
					}
				}
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
