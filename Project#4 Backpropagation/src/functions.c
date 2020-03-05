/*
 *	Further functions 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <utils.h>
#include <pthread.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

void generate(struct DATA *data, int datasize, int M, int N){
	
	/* Function that reads the data of training examples from a specific file */
	
	FILE *fpInput, *fpOutput;
	fpInput = fopen("../data/input10000x784float.txt","r");
		if (fpInput == NULL){
			printf("ERROR: The file input did not open successfully \n");
			exit(1);
		}
		
	fpOutput = fopen("../data/output10000.v","r");
	if (fpOutput == NULL){
		printf("ERROR: The file input did not open successfully \n");
		exit(1);
	}	
	
	for(int i=0 ; i<datasize ; i++){
		data[i].input=(int *)malloc(M*sizeof(int)); //malloc for the the data's input array 
		if (data[i].input == NULL){
			printf("\n ERROR: generate.1.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		data[i].teach=(int *)calloc(N, sizeof(int)); //malloc for the the data's input array 
		if (data[i].teach == NULL){
			printf("\n ERROR: generate.1.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		int j=0;
		fscanf (fpInput, "%d", &data[i].input[j]);    
		while ((!feof (fpInput)) && (j<M-1))
		{  
			
			j++;
			fscanf (fpInput, "%d", &data[i].input[j]);    			
		}
		
		int num;
		fscanf (fpOutput, "%d", &num); 
		data[i].teach[num]=1;
		
	}
	
	fclose(fpInput);	
	fclose(fpOutput);	
}

__global__ void cuda_init(unsigned int seed, curandState_t *states)
{
	/* Initializing the values of states */
	
	curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);

}

struct generate_data_threads{ // struct fot the Pthreads
	double **bias;
	double **weights;
	int threadChunk;
	int *M;
	int *N;
	double r;
	int* myLayers;
};

__global__ void cudaGenerator(double *weights_D, double *bias_D, int *M_D, double *r_D, curandState_t* statesB, curandState_t* statesW){
	int i=threadIdx.x, m= *M_D ;
	double r=*r_D;	
	
	/* RandomGeneratorB function: generates the values of biases between 0.0 ~ 1.0 */
	
	bias_D[i] = (double)curand_uniform(&statesB[i]); 
		
	for(int k=0 ; k < m; k++){
		
		/* RandomGeneratorB function: generates the values of weights between -r ~ r */
		
		weights_D[i*m + k] =((double)curand_uniform(&statesW[i*m + k])*2-1)*r; // generating the weights' values
			
	}
		

}

void *generate_data_parallel(void *thread_data){
	
	/* Function that calls the cuda kernel in parallel for all the network's layers */

	struct generate_data_threads *temp = (struct generate_data_threads *) thread_data;
	
	for(int j=0 ; j<temp->threadChunk ; j++){
		
		curandState_t *statesB, *statesW;
		cudaMalloc((void **)&statesB, temp->N[j]*sizeof(curandState_t));
		cudaMalloc((void **)&statesW, temp->N[j]*temp->M[j]*sizeof(curandState_t));
		cuda_init<<<1,temp->N[j]>>>(time(0)+100*temp->myLayers[j], statesB);
		cuda_init<<<1,(temp->N[j]*temp->M[j])>>>(time(0)+10*temp->myLayers[j], statesW);
			
		double *weights_D, *bias_D, *r_D;
		int *M_D;
		cudaMalloc((void **)&weights_D, temp->M[j]*temp->N[j]*sizeof(double));
		cudaMalloc((void **)&bias_D, temp->N[j]*sizeof(double));
		cudaMalloc((void **)&M_D, sizeof(int));
		cudaMalloc((void **)&r_D, sizeof(double));
		
		cudaMemcpy(M_D, &(temp->M[j]),sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(r_D, &(temp->r),sizeof(double), cudaMemcpyHostToDevice);
		
		cudaGenerator<<<1,temp->N[j]>>>(weights_D, bias_D, M_D, r_D, statesB, statesW);
		
		cudaMemcpy(temp->weights[j], weights_D, temp->M[j]*temp->N[j]*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(temp->bias[j], bias_D,temp->N[j]*sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(weights_D);
		cudaFree(M_D);
		cudaFree(r_D);
		cudaFree(bias_D);	
		cudaFree(statesB);
		cudaFree(statesW);
			
	}
	
	pthread_exit(NULL);
}

void generate_data(int numberOfThreads, double **weights, double **bias, int numberOfLayers, int *neuronsOfLayer){
	
	/* Function that generates the weight and bias array using pThreads and CUDA in a hybrid way */

	double r= sqrt(1.0/(double)(neuronsOfLayer[0])); // calculating the deviation for the weights' values
	
	int threadCheck, chunk, i;
	void *status;
	
	pthread_t threads[numberOfThreads];
	pthread_attr_t joinable;
	pthread_attr_init(&joinable);
	pthread_attr_setdetachstate(&joinable, PTHREAD_CREATE_JOINABLE);

	chunk = (numberOfLayers-1)/numberOfThreads;
	if ( chunk == 0 ) chunk = 1;
	
	struct generate_data_threads thread_data[numberOfThreads];
	int sum = 0, t=0;
	
	for ( i=0 ; i<numberOfThreads ; i++){
		thread_data[i].threadChunk = chunk;
		if ( i < numberOfThreads-1 ) sum+= chunk;					
		if ( i == numberOfThreads-1 ) thread_data[i].threadChunk = numberOfLayers-1-sum;
		
		thread_data[i].bias= (double **)malloc(thread_data[i].threadChunk*sizeof(double*)); // malloc for bias array
		if (thread_data[i].bias == NULL){
			printf("\n ERROR: thread_data[i].bias.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		thread_data[i].M= (int *)malloc(thread_data[i].threadChunk*sizeof(int)); // malloc for M array
		if (thread_data[i].M == NULL){
			printf("\n ERROR: thread_data[i].M.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		thread_data[i].N= (int *)malloc(thread_data[i].threadChunk*sizeof(int)); // malloc for N array
		if (thread_data[i].N == NULL){
			printf("\n ERROR: thread_data[i].N.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		thread_data[i].myLayers= (int *)malloc(thread_data[i].threadChunk*sizeof(int)); // malloc for myLayers array
		if (thread_data[i].myLayers == NULL){
			printf("\n ERROR: thread_data[i].myLayers.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		thread_data[i].weights= (double **)malloc(thread_data[i].threadChunk*sizeof(double*)); // malloc for weights array
		if (thread_data[i].weights == NULL){
			printf("\n ERROR: thread_data[i].weights.Memory allocation did not complete successfully!\n");
			exit (1);
		}
		
		thread_data[i].r = r;
		
		for (int j=0 ; j<thread_data[i].threadChunk ; j++){
			thread_data[i].N[j] = neuronsOfLayer[1 + t];
			thread_data[i].M[j] = neuronsOfLayer[t];
			thread_data[i].bias[j] = bias[t];
			thread_data[i].weights[j] = weights[t];		
			thread_data[i].myLayers[j] = t;	
			t++;
		}	
		threadCheck = pthread_create(&threads[i], &joinable,generate_data_parallel , (void *)&thread_data[i]);

		if(threadCheck){
			printf("ERROR: pthread_create() returned code %d \n", threadCheck);
			return;
		}
	}
	
	pthread_attr_destroy(&joinable); // free attribute and wait for the other threads
	
	for ( i=0 ; i<numberOfThreads ; i++){
		
		threadCheck = pthread_join(threads[i], &status);
		 if (threadCheck)
        {
             printf("ERROR; return code from pthread_join() is %d\n", threadCheck);
			 exit(1);
        }
	}
	
}