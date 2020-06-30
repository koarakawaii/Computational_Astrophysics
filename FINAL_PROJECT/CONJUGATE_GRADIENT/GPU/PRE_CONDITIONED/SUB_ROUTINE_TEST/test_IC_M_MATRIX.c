#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void PRE_CONDITION_IC(int, double, double, double*, double*, double*);
void PRODUCE_CHOLESKY(int, int, double, double, double*);


int main(void)
{
	printf("Test the forward and backward substitution for IC pre-condition.\n");
	
	int N, row;
	double dx, photon_mass;
	double *x, *R, *temp, *x_prime, *x_prime_sol;
	FILE* output;
	
	printf("Set the dimension of lattice (N,N).\n");
	scanf("%d",&N);
	printf("The dimension of the lattice is set as (%d,%d) .\n",N,N);
	printf("Set the photon mass.\n");
	scanf("%lf", &photon_mass);
	printf("The photon mass is set as %.4e .\n", photon_mass);
	printf("\n");

	dx = 1./(N-1);
	row =N*N;
	x = malloc(N*N*sizeof(double));
	R = malloc(N*N*3*sizeof(double));
	temp = malloc(N*N*sizeof(double));
	x_prime = malloc(N*N*sizeof(double));
	x_prime_sol = malloc(N*N*sizeof(double));
	srand(time(NULL));
	output = fopen("x_prime_IC.txt","w");
	
	for (int i=0; i<N*N; i++)
		x_prime[i] = (double)(rand())/(double)(RAND_MAX);

	PRODUCE_CHOLESKY(N, row, dx, photon_mass, R);

	for (int i=0; i<N*N; i++)
	{
		int i_x = i%N;
		int i_y = i/N;
		
		if (i_x!=0&&i_x!=N-1&&i_y!=0&&i_y!=N-1)
		{
			if (i_x<N-2&&i_y<N-2) 
				temp[i] = R[3*i+2]*x_prime[i+N] + R[3*i+1]*x_prime[i+1] + R[3*i]*x_prime[i];
			else if (i_x<N-2)
				temp[i] = R[3*i+1]*x_prime[i+1] + R[3*i]*x_prime[i];
			else if (i_y<N-2)
				temp[i] = R[3*i+2]*x_prime[i+N] + R[3*i]*x_prime[i];
			else 
				temp[i] = R[3*i]*x_prime[i];
		}
		else
			temp[i] = x_prime[i];
//		fprintf(output,"%.8e\t%.8e\n",x_prime[i], temp[i]);
	}
	for (int i=0; i<N*N; i++)
	{
		int i_x = i%N;
		int i_y = i/N;
		
		if (i_x!=0&&i_x!=N-1&&i_y!=0&&i_y!=N-1)
		{
			if (i_x>1&&i_y>1)
				x[i] = R[3*(i-N)+2]*temp[i-N] + R[3*(i-1)+1]*temp[i-1] + R[3*i]*temp[i];
			else if (i_x>1)
				x[i] = R[3*(i-1)+1]*temp[i-1] + R[3*i]*temp[i];
			else if (i_y>1)
				x[i] = R[3*(i-N)+2]*temp[i-N] + R[3*i]*temp[i];
			else
				x[i] = R[3*i]*temp[i];
		}
		else
			x[i] = temp[i];
//		printf("x_prime[%d] = %.8e ;\tx[%d] = %.8e \n",i, x_prime[i], i, x[i]);
		fprintf(output,"%.8e\t%.8e\n",x_prime[i], x[i]);
	}

//	printf("\n");
	printf("Start inverse matrix M...\n");
	clock_t start = clock();
	PRE_CONDITION_IC(N, dx, photon_mass, R,	x, x_prime_sol);
	clock_t end = clock();

	double error = 0.0;
	double norm = 0.0;
	for (int i=0; i<N*N; i++)
	{
//		printf("x_prime[%d] = %.8e ;\tx_prime_sol[%d] = %.8e \n",i, x_prime[i], i, x_prime_sol[i]);
		norm += x_prime[i]*x_prime[i];
		error += pow(x_prime_sol[i]-x_prime[i], 2.);
	}
	printf("\n");
	printf("Error is %.16e ; total %.4f ms is taken.\n", sqrt(error/norm), 1000*(double)(end-start)/CLOCKS_PER_SEC);

	free(x);
	free(temp);
	free(x_prime);
	free(x_prime_sol);
	return EXIT_SUCCESS;
}

void PRE_CONDITION_IC(int N, double dx, double photon_mass,double* R, double* r, double* r_prime)
{
    for (int idx=0; idx<N*N; idx++)
    {
        int idx_x = idx%N;
        int idx_y = idx/N;
        if ( idx_x!=0 && idx_x!=N-1 && idx_y!=0 && idx_y!=N-1 )
		{
			if (idx_x>1&&idx_y>1)
				r_prime[idx] = (r[idx]-(R[3*(idx-1)+1]*r_prime[idx-1]+R[3*(idx-N)+2]*r_prime[idx-N]))/R[3*idx];
			else if (idx_x>1)
				r_prime[idx] = (r[idx]-R[3*(idx-1)+1]*r_prime[idx-1])/R[3*idx];
			else if (idx_y>1)
				r_prime[idx] = (r[idx]-R[3*(idx-N)+2]*r_prime[idx-N])/R[3*idx];
			else
				r_prime[idx] = r[idx]/R[3*idx];
		}
        else
            r_prime[idx] = r[idx];
//		printf("r_prime[%d]\t%.8f\n", idx, r_prime[idx]);
    }                                                                     
	for (int idx=N*N-1; idx>=0; idx--)
	{
        int idx_x = idx%N;
        int idx_y = idx/N;
        if ( idx_x!=0 && idx_x!=N-1 && idx_y!=0 && idx_y!=N-1 )
		{
			if (idx_x<N-2&&idx_y<N-2)
				r_prime[idx] = (r_prime[idx]-(R[3*idx+1]*r_prime[idx+1]+R[3*idx+2]*r_prime[idx+N]))/R[3*idx];
			else if (idx_x<N-2)
				r_prime[idx] = (r_prime[idx]-R[3*idx+1]*r_prime[idx+1])/R[3*idx];
			else if (idx_y<N-2)
				r_prime[idx] = (r_prime[idx]-R[3*idx+2]*r_prime[idx+N])/R[3*idx];	
			else
				r_prime[idx] = r_prime[idx]/R[3*idx];
		}
	}
}

void PRODUCE_CHOLESKY(int N, int row, double dx, double photon_mass, double* R)
{
	int idx_x, idx_y;
	double temp;

	for (int idx=0; idx<N+1; idx++)
		R[3*idx] = 1.0;	//diagonal
	for (int idx=N+1; idx<row-N-1; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			R[3*idx] = (4.+pow(photon_mass*dx,2.));
			if (idx_x<N-2)
				R[3*idx+1] = -1.;
			if (idx_y<N-2)
				R[3*idx+2] = -1.;
		}
		else
			R[3*idx] = 1.0;
	}
	for (int idx=row-N-1; idx<row; idx++)
		R[3*idx] = 1.0;

	for (int idx=N+1; idx<row-N-1; idx++)
	{
		idx_x = idx%N;
		idx_y = idx/N;
		
		if (idx_x!=0&&idx_x!=N-1&&idx_y!=0&&idx_y!=N-1)
		{
			temp = sqrt(R[3*idx]);
			R[3*idx] = temp;
			if (idx_x<N-2)
			{
				R[3*idx+1] /= temp;
				R[3*idx+3] -= pow(R[3*idx+1],2.);
			}
			if (idx_y<N-2)
			{
				R[3*idx+2] /= temp;
				R[3*(idx+N)] -= pow(R[3*idx+2],2.);
			}
		}
	}
}
