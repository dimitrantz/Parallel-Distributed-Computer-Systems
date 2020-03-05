#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <omp.h>

#define DIM 3


void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){


  extern int numThreads; 
  int i=0;

  #pragma omp parallel num_threads(numThreads) private(i) shared(Y, X, permutation_vector)
  {

     #pragma omp for schedule(static) //schedule is defined as static due to the equivalent work of each thread
		for(i=0; i<N; i++){
			memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
	    }
     
  }
}
