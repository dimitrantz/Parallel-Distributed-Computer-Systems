#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"

#define DIM 3

inline unsigned long int splitBy3(unsigned int a){
    unsigned long int x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

struct morton_threads{ // struct with which we pass multiple arguments in the pthread_create() routine 
	unsigned long int *mcodes;
	unsigned int *codes;
	int threadsChunk;
};


inline unsigned long int mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
    unsigned long int answer;
    answer = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

	/*
	Creation of parallel morton. 
	This code below is being parallelized through this routine that each thread will execute once it is created
	 for(int i=0; i<N; i++){
		  // Compute the morton codes from the hash codes using the magicbits mathod
		 mcodes[i] = mortonEncode_magicbits(codes[i*DIM], codes[i*DIM + 1], codes[i*DIM + 2]);
	}
	*/


void  *parallel_morton(void * thread_data) 
{
	struct morton_threads *temp= ( struct morton_threads *) thread_data; // typecasting the void struct to morton_threads struct
	
	int i;
	for(i=0; i<temp->threadsChunk; i++)
	{ 
	    temp->mcodes[i] = mortonEncode_magicbits(temp->codes[i*DIM], temp->codes[i*DIM + 1], temp->codes[i*DIM + 2]);
	}
	 pthread_exit(NULL);
}



/* The function that transform the morton codes into hash codes */ 
void morton_encoding(unsigned long int *mcodes, unsigned int *codes, int N, int max_level){

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

	struct morton_threads thread_data[numThreads];

	for(i = 0; i < numThreads; i++)
    { //Each thread receives a unique instance of the structure
        thread_data[i].mcodes = &mcodes[chunk*i];
		thread_data[i].codes = &codes[DIM*chunk*i];
        thread_data[i].threadsChunk = chunk;
		
		if (i<numThreads-1)
			sum+=chunk;

		if (i==numThreads-1) // the last thread will execute the remaining work
			thread_data[i].threadsChunk = N-sum;


        threadCheck = pthread_create(&threads[i], &joinable, parallel_morton, (void *)&thread_data[i]);

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

  



