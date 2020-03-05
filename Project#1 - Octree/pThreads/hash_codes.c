#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "pthread.h"

#define DIM 3

inline unsigned int compute_code(float x, float low, float step){

  return floor((x - low) / step);

}

struct hash_threads{// struct with which we pass multiple arguments in the pthread_create() routine 
	unsigned int *codes;
	float *X;
	float *low;
	float step;
	int threadsChunk;
};


/*
	Creation of parallel quantize. 
	This code below is being parallelized through this routine that each thread will execute once it is created
	  for(int i=0; i<N; i++){
        for(int j=0; j<DIM; j++){
		 codes[i*DIM + j] = compute_code(X[i*DIM + j], low[j], step); 
		}
	}
*/


void  *parallel_quantize(void * thread_data)
{
	struct hash_threads *temp= ( struct hash_threads *) thread_data; // typecasting the void struct to hash_threads struct
	int i;
	for( i=0; i<temp->threadsChunk; i++)
	{
	  for(int j=0; j<DIM; j++)
	  {
	      temp->codes[i*DIM + j] = compute_code(temp->X[i*DIM + j], temp->low[j], temp->step); 
	  }
	}
	 pthread_exit(NULL);

}


/* Function that does the quantization */
void quantize(unsigned int *codes, float *X, float *low, float step, int N){

	extern int numThreads;
	int i, threadCheck;
	float sum=0;
	void *status;

	pthread_t threads[numThreads]; //array of threads' identities
	pthread_attr_t joinable; // declare a pthread attribure
	pthread_attr_init(&joinable); // initialize the threads' attribute as joinable
	pthread_attr_setdetachstate(&joinable, PTHREAD_CREATE_JOINABLE); //determines that all threads created are joinable

	int chunk=N/numThreads;
	if (chunk==0) 
		chunk=1;

	struct hash_threads thread_data[numThreads];

	for(i = 0; i < numThreads; i++)
    { //Each thread receives a unique instance of the structure
        thread_data[i].codes = &codes[DIM*chunk*i];
        thread_data[i].X = &X[DIM*chunk*i];
        thread_data[i].low = low;
        thread_data[i].step = step;
        thread_data[i].threadsChunk = chunk;
		
		if (i<numThreads-1)
			sum+=chunk;

		if (i==numThreads-1) // the last thread will execute the remaining work
			thread_data[i].threadsChunk = N-sum;


        threadCheck = pthread_create(&threads[i], &joinable, parallel_quantize, (void *)&thread_data[i]);

        if(threadCheck)
        {
            printf("Error: pthread_create returned code %d\n", threadCheck);
            return;
        }
    }

	pthread_attr_destroy(&joinable); // free attribute and wait for the other threads
	for (i=0;i<numThreads;i++){
		threadCheck = pthread_join(threads[i], &status);
		 if (threadCheck)
        {
             printf("ERROR; return code from pthread_join() is %d\n", threadCheck);
			 exit(-1);
        }
	}
}

float max_range(float *x){

  float max = -FLT_MAX;
  for(int i=0; i<DIM; i++){
    if(max<x[i]){
      max = x[i];
    }
  }

  return max;

}

void compute_hash_codes(unsigned int *codes, float *X, int N, 
			int nbins, float *min, 
			float *max){
  
  float range[DIM];
  float qstep;

  for(int i=0; i<DIM; i++){
    range[i] = fabs(max[i] - min[i]); // The range of the data
    range[i] += 0.01*range[i]; // Add something small to avoid having points exactly at the boundaries 
  }

  qstep = max_range(range) / nbins; // The quantization step 
  
  quantize(codes, X, min, qstep, N); // Function that does the quantization

}



