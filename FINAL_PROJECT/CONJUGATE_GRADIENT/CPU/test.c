#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <gsl/gsl_rng.h>
#include <string.h>
#include <math.h>

int main(void)
{
	int N;
	double sum_1, sum_2;
	long seed;
	double *A, *B;
	gsl_rng *rng;

	printf("CBLAS test...\n\n");
	printf("Set the dimension N for the array A, B, and C.\n");
	scanf("%d", &N);
	printf("The dimension of the array is %d .\n", N);
	printf("Set the seed for random number generator.\n");
	scanf("%ld", &seed);
	printf("The seed number of random number generator is %ld .\n", seed);
	printf("\n");
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, seed);
	
	A = malloc(N*sizeof(double));
	B = malloc(N*sizeof(double));

	printf("Test the DDOT.\n");
	sum_1 = 0.0;

	for (int i=0; i<N; i++)
	{
		A[i] = gsl_rng_uniform(rng);
		B[i] = gsl_rng_uniform(rng);
		sum_1 += A[i]*B[i];
	}
	printf("The dot of A and B calculate by for loop is %.16f .\n", sum_1);

	int inc = 1;
	sum_2 = cblas_ddot(N, A, inc, B, inc);
	printf("The dot of A and B calculate by CBLAS is %.16f .\n", sum_2);

	printf("The error between for loop and CBLAS is %.16f .\n", (sum_1-sum_2)/sum_1);

	printf("===================================================\n");
	printf("Test the SAXPY.\n");
	
	double alpha;
	printf("Set the value of alpha.\n");
	scanf("%lf", &alpha);
	printf("The value of alpha is %.6f .\n", alpha);
	
	double *C_1 = malloc(N*sizeof(double));
	double *C_2 = malloc(N*sizeof(double));
	memcpy(C_2, B, N*sizeof(double));

	for (int i=0; i<N; i++)
		C_1[i] = alpha*A[i]+B[i];

	cblas_daxpy(N, alpha, A, 1, C_2, 1);
	double error = 0.0, norm = 0.0;
	for (int i=0; i<N; i++)
	{
		error += pow(C_2[i]-C_1[i],2);
		norm += pow(C_1[i],2);
	}
	error /= norm;
	error = sqrt(error);
	printf("The error between DAXPY done by for loop and CBLAS is %.16f .\n", error);
	printf("===================================================\n");

	free(A);
	free(B);
	free(C_1);
	free(C_2);
	return EXIT_SUCCESS;
}

