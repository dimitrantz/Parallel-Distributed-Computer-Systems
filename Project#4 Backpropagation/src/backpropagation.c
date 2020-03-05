/*
 *	Backpropagation algorithm
 *	 firstly, it is needed to transpose the array of the weights for each layer in order to multiply
 *	 it with the layer's array of delta. Moving now from the end of the network to its start, the delta 
 *	 for each neuron of the network is cumputed  
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <cuda.h>


__global__ void cudaBackpropagation(int *N, int *M, double *weights_D, double *delta_D, double *deltaJ_D, double *z_D, double *temp_D)
{ 	
	/* CudaBackpropagation: this cuda kernel implements the computation of delta values for all the neurons of a layer in parallel, using the nnet's weights */
	
	int k = blockIdx.x*blockDim.x + threadIdx.x;
	int m = *M;
	int n = *N;
	double sum = 0.0;
	
	for(int i=0 ; i<m ; i++){ 
	
		temp_D[i*n + k] = weights_D[k*m + i]; // implement the transpose
		sum += temp_D[i*n + k]*delta_D[i];	
		
	}
	
	deltaJ_D[k] = ((double)1)/(double)(1+ exp(-z_D[k]))*(1-((double)1)/(double)(1+ exp(-z_D[k])))*sum; // calculate the array of deltas	
	
}

void launchKernelB(int numberOfLayers, int *neuronsOfLayer, double **weights, double **delta, double **z)
{
	/* launchKernelB: calls the cuda kernel cudaBackpropagation for all the nnet's layers for all their neurons, except from the input layer */
	
	for(int j=numberOfLayers-2 ; j>0 ; j--){
		
		double *temp; // temporary array, used for the transposed array of weigts
		temp = (double *)malloc((neuronsOfLayer[j]*neuronsOfLayer[j+1]*sizeof(double))); // malloc for the temporary array
		if (temp == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully for the transpose!\n");
			exit (1);
		}
	
  
		double *temp_D, *weights_D, *delta_D, *deltaJ_D, *z_D;
		int *M, *N;
		
		cudaMalloc((void **)&temp_D, neuronsOfLayer[j+1]*neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&weights_D, neuronsOfLayer[j+1]*neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&delta_D, neuronsOfLayer[j+1]*sizeof(double));
		cudaMalloc((void **)&deltaJ_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&z_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&M, sizeof(int));
		cudaMalloc((void **)&N, sizeof(int));
		
		cudaMemcpy(weights_D, weights[j], neuronsOfLayer[j+1]*neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(temp_D, temp, neuronsOfLayer[j+1]*neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(z_D, z[j-1], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(delta_D, delta[j], neuronsOfLayer[j+1]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(M, &neuronsOfLayer[j+1], sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(N, &neuronsOfLayer[j], sizeof(int), cudaMemcpyHostToDevice);
			
		cudaBackpropagation<<<1,neuronsOfLayer[j]>>>(N, M, weights_D, delta_D, deltaJ_D, z_D, temp_D);

		cudaMemcpy(delta[j-1], deltaJ_D, neuronsOfLayer[j]*sizeof(double), cudaMemcpyDeviceToHost);
		
		free(temp); // free the transposed array of weights
		
		cudaFree(temp_D);
		cudaFree(weights_D);
		cudaFree(delta_D);
		cudaFree(deltaJ_D);
		cudaFree(N);
		cudaFree(M);
		cudaFree(z_D);
	}
	
}
