
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pthread.h"

#define DIM 3

struct dataRear_threads{ // struct with which we pass multiple arguments in the pthread_create() routine 
	float *Y;
	float *X;
	unsigned int *permutation_vector;
	int threadsChunk;
};


/*
	Creation of parallel dataRear. 
	This code below is being parallelized through this routine that each thread will execute once it is created
	  for(int i=0; i<N; i++){
		 memcpy(&Y[i*DIM], &X[permutation_vector[i]*DIM], DIM*sizeof(float));
	  }
*/

void  *parallel_dataRear(void * thread_data)
{
	struct dataRear_threads *temp= ( struct dataRear_threads *) thread_data; // typecasting the void struct to morton_threads struct
	
	int i;
	for(i=0; i<temp->threadsChunk; i++)
	{ 
	     memcpy(&temp->Y[i*DIM], &temp->X[temp->permutation_vector[i]*DIM], DIM*sizeof(float));
	}

	 pthread_exit(NULL);
}



void data_rearrangement(float *Y, float *X, 
			unsigned int *permutation_vector, 
			int N){

	extern int numThreads;
    int i, threadCheck, sum=0;
	void *status;

    pthread_t threads[numThreads]; //array of threads' identities
	pthread_attr_t joinable; // declare a pthread attribure
	pthread_attr_init(&joinable); // initialize the threads' attribute as joinable
	pthread_attr_setdetachstate(&joinable, PTHREAD_CREATE_JOINABLE); //determines that all threads created are joinable

	int chunk=N/numThreads;
	if (chunk==0) 
		chunk=1;

	struct dataRear_threads thread_data[numThreads];

	for(i = 0; i < numThreads; i++)
    { //Each thread receives a unique instance of the structure
        thread_data[i].X = X;
		thread_data[i].Y = &Y[DIM*chunk*i];
		thread_data[i].permutation_vector=&permutation_vector[chunk*i];
		
		if (i==numThreads-1) // the last thread will execute the remaining work
			{
			thread_data[i].threadsChunk = N-sum;
			}
			else{
        thread_data[i].threadsChunk = chunk;
		}
		
		if (i<numThreads-1)
			sum+=chunk;
			
        threadCheck = pthread_create(&threads[i], &joinable, parallel_dataRear, (void *)&thread_data[i]);

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
