/*
 *	Update weights algorithm
 *  at this point it is needed to calculate the derivatives of the loss function.  
 *	In other words, the values with which the weights will be updated are computed here 
 */
 
#include <utils.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void cudaUpdateWeights(int *M, double *eta_D, int *datasize_D, double *weights_D, double *delta_D, double *activation_D, double *bias_D)
{
	/* CudaUpdateWeights: this cuda kernel updates the weights for all the neurons of a layer in parallel, using each layer's activation  */
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int m = *M;
	int datasize = *datasize_D;;
	double eta = *eta_D;
		
	for(int k=0; k<m ; k++){
		weights_D[i*m + k] -= (eta/(double)datasize)*(delta_D[i] * activation_D[k]); // update the weights 
	}	
	bias_D[i] -= (eta/(double)datasize)*delta_D[i]; // update the biases 

}


void launchKernelU(double eta, int datasize, double **weights, double **delta, double **activation, int numberOfLayers, int *neuronsOfLayer, double **bias)
{
	/* launchKernelU: calls the cuda kernel cudaUpdateWeights for all the nnet's layers for all their neurons, except from the input layer */
	
	for(int j=1 ; j<numberOfLayers ; j++){
	
		double *weights_D, *activation_D, *eta_D, *bias_D, *delta_D;
		int *M, *datasize_D;
	
		cudaMalloc((void **)&bias_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&weights_D, neuronsOfLayer[j-1]*neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&activation_D, neuronsOfLayer[j-1]*sizeof(double));
		cudaMalloc((void **)&delta_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&eta_D, sizeof(double));
		cudaMalloc((void **)&M, sizeof(int));
		cudaMalloc((void **)&datasize_D, sizeof(int));
		
		cudaMemcpy(weights_D, weights[j-1], neuronsOfLayer[j-1]*neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(bias_D, bias[j-1], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(delta_D, delta[j-1], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(activation_D, activation[j-1], neuronsOfLayer[j-1]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(M, &neuronsOfLayer[j-1], sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(eta_D, &eta, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(datasize_D, &datasize, sizeof(int), cudaMemcpyHostToDevice);
		
		
		cudaUpdateWeights<<<1,neuronsOfLayer[j]>>>(M, eta_D, datasize_D, weights_D, delta_D, activation_D, bias_D );

		cudaMemcpy(weights[j-1], weights_D, neuronsOfLayer[j]*neuronsOfLayer[j-1]*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(bias[j-1], bias_D, neuronsOfLayer[j]*sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaFree(bias_D);
		cudaFree(weights_D);
		cudaFree(activation_D);
		cudaFree(delta_D);
		cudaFree(M);
		cudaFree(eta_D);
		cudaFree(datasize_D);		
	}
}
