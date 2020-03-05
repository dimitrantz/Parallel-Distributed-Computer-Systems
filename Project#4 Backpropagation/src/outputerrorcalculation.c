/*
 *	Output error calculation algorithm
 * 	at this point, the difference between the output of the network and the expected output is calculated 
 *	for each neuron. Then, it is stored as delta for each neuron of the last network's layer 	   
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <cuda.h>


__global__ void cudaOutputerrorcalculation(double *errorArray_D, int *teach_D, double *delta_D, double *activation_D,  double *z_D )
{
	/* CudaOutputerrorcalculation: this cuda kernel computes the delta values for all the neurons of the output layer in parallel, using z values  */
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	delta_D[i] = activation_D[i]-(double)teach_D[i]; // partial derivative of the loss function, value a 
	
	errorArray_D[i] = delta_D[i];
	errorArray_D[i] *= errorArray_D[i]/2;
	
	delta_D[i] *= ((double)1)/(double)(1+ exp(-z_D[i]))*(1-((double)1)/(double)(1+ exp(-z_D[i]))); 
}

void launchKernelE(double *errorArray, int numberOfLayers, int *neuronsOfLayer, int *teach, double **delta, double **activation, double **z)
{
	/* launchKernelE: calls the cuda kernel cudaOutputerrorcalculation for the output layer */
	
	double *errorArray_D, *activation_D, *delta_D, *z_D ;
	int *teach_D;
	
	cudaMalloc((void **)&errorArray_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double));
	cudaMalloc((void **)&teach_D, neuronsOfLayer[numberOfLayers-1]*sizeof(int));
	cudaMalloc((void **)&delta_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double));
	cudaMalloc((void **)&activation_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double));
	cudaMalloc((void **)&z_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double));
	
	
	cudaMemcpy(teach_D, teach, neuronsOfLayer[numberOfLayers-1]*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(delta_D, delta[numberOfLayers-2], neuronsOfLayer[numberOfLayers-1]*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(activation_D, activation[numberOfLayers-1], neuronsOfLayer[numberOfLayers-1]*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(z_D, z[numberOfLayers-2], neuronsOfLayer[numberOfLayers-1]*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaOutputerrorcalculation<<<1,neuronsOfLayer[numberOfLayers-1]>>>(errorArray_D, teach_D, delta_D, activation_D, z_D);
	
	cudaMemcpy(errorArray, errorArray_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(delta[numberOfLayers-2], delta_D, neuronsOfLayer[numberOfLayers-1]*sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(errorArray_D);
	cudaFree(teach_D);
	cudaFree(delta_D);
	cudaFree(activation_D);
	cudaFree(z_D);

}

