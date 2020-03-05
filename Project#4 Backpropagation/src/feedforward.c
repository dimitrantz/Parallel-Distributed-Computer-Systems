/*
 *	Feedforward algorithm
 *  in this step, the algorithm moves from the beginning of the network to the end,
 *  in order to compute activation value, which is the result of the sigmoid function σ(z)
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <cuda.h>


__global__ void cudafeedforward(int *M, double *weights_D, double *activation_D, double *activationJ_D, double *bias_D, double *z_D)
{
/* Cudafeedforward: this cuda kernel implements the computation of z and activation values for all the neurons of a layer in parallel */
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int m = *M;

	z_D[i]=0.0;
	for(int k=0 ; k<m ; k++){  
		z_D[i]+= weights_D[i*m + k]*activation_D[k]; 	
	}
	z_D[i] += bias_D[i]; // calculating z for each neuron
	activationJ_D[i]  = ((double)1)/(double)(1+ exp(-z_D[i])); // calculating activation for each neuron, a=σ(z)
	
}


void launchKernelF(int numberOfLayers, int *neuronsOfLayer, double **weights, double **activation, double **bias, double **z)
{
	/* launchKernelF: calls the cuda kernel cudafeedforward for all the nnet's layers for all their neurons, except from the input layer */
	
	for(int j=1 ; j<numberOfLayers ; j++){
		double *weights_D, *activation_D, *activationJ_D, *z_D, *bias_D;
		int *M;
	
		cudaMalloc((void **)&bias_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&weights_D, neuronsOfLayer[j-1]*neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&activation_D, neuronsOfLayer[j-1]*sizeof(double));
		cudaMalloc((void **)&activationJ_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&z_D, neuronsOfLayer[j]*sizeof(double));
		cudaMalloc((void **)&M, sizeof(int));
		
		cudaMemcpy(weights_D, weights[j-1], neuronsOfLayer[j-1]*neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(bias_D, bias[j-1], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(z_D, z[j-1], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(activation_D, activation[j-1], neuronsOfLayer[j-1]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(activationJ_D, activation[j], neuronsOfLayer[j]*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(M, &neuronsOfLayer[j-1] , sizeof(int), cudaMemcpyHostToDevice);	
		
		cudafeedforward<<<1,neuronsOfLayer[j]>>>(M, weights_D, activation_D, activationJ_D, bias_D, z_D); 

		cudaMemcpy(activation[j], activationJ_D, neuronsOfLayer[j]*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z[j-1], z_D, neuronsOfLayer[j]*sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaFree(bias_D);
		cudaFree(weights_D);
		cudaFree(activation_D);
		cudaFree(activationJ_D);
		cudaFree(M);
		cudaFree(z_D);
	}
}
