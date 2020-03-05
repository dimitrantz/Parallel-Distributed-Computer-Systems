/*
 *	Main function
 */

#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include "sys/time.h"

int main(int argc, char** argv)
{		

	// Time counting variables 
	struct timeval startwtime, endwtime;
	
	if (argc != 6) { // check if the command line arguments are correct 
		printf("Usage: %s N thres disp\n"
		"where\n"
		"eta : learning rate\n"
		"epochs : number of epochs of training\n"
		"datasize : training examples.\n" 
		"numberOfLayers : number of layers.\n" 
		"numberOfThreads : number of threads.\n" , argv[0]);
		return (1);
	}
	
	double eta = atof(argv[1]); // learning rate
	int epochs = atoi(argv[2]);	// number of epochs of training
	int datasize = atoi(argv[3]); // number of training samples
	int numberOfLayers = atoi(argv[4]); // number of network's layers
	int numberOfThreads = atoi(argv[5]);	
	int mode;
	
	if (numberOfLayers<3){ // check the number of layers entered, to be at least three
		printf("Wrong input. Layers should be at least 3. Try again! \n");
		exit (1);
	}
	
	do{
		printf(" Select the way of entering the number of neurons for each layer! \n"
		"Press 1 if you want to read from a file, otherwise press 0. \n");
		scanf("%d",&mode);
	}while(mode!=0 && mode!=1);
		
	int *neuronsOfLayer = (int *)malloc(numberOfLayers*sizeof(int)); // malloc for the neuronsOfLayer array
	if (neuronsOfLayer == NULL){
		printf("\nERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	if(mode == 0){
		for (int i=0 ; i<numberOfLayers ; i++){ // enter the number of neurons for each layer	
			do{
				printf("Enter the number of neurons of the %d th layer: \n", i);
				scanf("%d",&neuronsOfLayer[i]);
			}while(neuronsOfLayer[i]<=0);
		}
	}
	else{
		FILE *fp;
		fp = fopen("../data/data_neurons.txt","r");
		if (fp == NULL){
			printf("ERROR: The file data_neurons did not open successfully \n");
			exit(1);
		}
		int i=0;
		fscanf (fp, "%d", &neuronsOfLayer[i]);    
		while (!feof (fp))
		{  
			i++;
			fscanf (fp, "%d", &neuronsOfLayer[i]);    			
		}

		fclose(fp);		
	}
	
	double *errorArray, **weights, **bias, **z, **delta, **activation;
	
	errorArray = (double *)malloc(neuronsOfLayer[numberOfLayers-1]*sizeof(double)); // malloc for the errorArray array
	if (errorArray == NULL){
		printf("\nERROR: Memory allocation did not complete successfully for errorArray!\n");
		return (1);
	}
	
	weights = (double **)malloc((numberOfLayers-1)*sizeof(double*));  // malloc for weights array
	if (weights == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully for weights!\n");
		exit (1);
	}
	
	bias = (double **)malloc((numberOfLayers-1)*sizeof(double*));  // malloc for bias array
	if (bias == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully for bias!\n");
		exit (1);
	}
	
	z = (double **)malloc((numberOfLayers-1)*sizeof(double*)); // malloc for z array
	if (z == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully for z!\n");
		exit (1);
	}
	
	delta = (double **)malloc((numberOfLayers-1)*sizeof(double*)); // malloc for delta array
	if (delta == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully for delta!\n");
		exit (1);
	}
	
	activation = (double **)malloc(numberOfLayers*sizeof(double*)); // malloc for activation array
	if (activation == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully for activation!\n");
			exit (1);
	}
	
	activation[0] = (double *)malloc(neuronsOfLayer[0]*sizeof(double));
	if (activation[0] == NULL){
		printf("\n ERROR: activation[0].Memory allocation did not complete successfully for activation[0]!\n");
		exit (1);
	}
	
	for(int j=1 ; j<numberOfLayers ; j++){    
	                                              
		bias[j-1] = (double *)malloc(neuronsOfLayer[j]*sizeof(double));
		if (bias[j-1] == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully bias[j]!\n");
			exit (1);
		}
		
		z[j-1] = (double *)malloc(neuronsOfLayer[j]*sizeof(double)); 
		if (z[j-1] == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully z[j]!\n");
			exit (1);
		}
		
		delta[j-1] = (double *)malloc(neuronsOfLayer[j]*sizeof(double)); 
		if (delta[j-1] == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully delta[j]!\n");
			exit (1);
		}
		
		activation[j] = (double *)malloc(neuronsOfLayer[j]*sizeof(double)); 
		if (activation[j] == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully activation[j]!\n");
			exit (1);
		}
		
		weights[j-1]= (double *)malloc(neuronsOfLayer[j]*neuronsOfLayer[j-1]*sizeof(double)); 
		if (weights[j-1] == NULL){
			printf("\n ERROR: Memory allocation did not complete successfully for weights[j]!\n");
			exit (1);
		}
    }
	
	generate_data(numberOfThreads, weights, bias, numberOfLayers, neuronsOfLayer); // generating the weights and the biases of the network
	
	struct DATA *data; // array of datasets
	data = (struct DATA *)malloc(datasize*sizeof(struct DATA)); // malloc for the data array 
	if (data == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully for data!\n");
		exit (1);
	}
	
	generate(data, datasize, neuronsOfLayer[0], neuronsOfLayer[numberOfLayers-1]); // generating the datasets
	
	/* training the network using the backpropagation algorithm */
	
	int counterEpochs = 0; // counter of the epochs

	double time = 0.0; // time variable
	
	while(counterEpochs < epochs){ 
	
		gettimeofday (&startwtime, NULL); // start counting time
		double error=0.0; // error variable
		printf("counterEpochs %d: \n", counterEpochs);
		
		for (int m=0 ; m<datasize ; m++){	
		
			for(int i=0 ; i<neuronsOfLayer[0] ; i++){
				activation[0][i]= (double)(data[m].input[i]);	// update the activation array of the input layer		
			}
			
			launchKernelF(numberOfLayers, neuronsOfLayer, weights, activation, bias, z);		
				
			launchKernelE(errorArray, numberOfLayers, neuronsOfLayer, data[m].teach, delta, activation,z);		
			for(int i=0 ; i<neuronsOfLayer[numberOfLayers-1] ; i++){
				error+= errorArray[i];
				
			}	
		
			launchKernelB(numberOfLayers, neuronsOfLayer, weights, delta, z);	
			
			launchKernelU(eta, datasize, weights, delta, activation, numberOfLayers, neuronsOfLayer, bias);
			
		}
		
		error /= 10*datasize; // 10 represents the number of neurons of the output layer
		
		if (error <0.045) {
			
			printf("The algorithm stopped in counterEpochs %d for error: %f\n", counterEpochs, error);
			
			gettimeofday (&endwtime, NULL);
			time += (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
			printf("Total time        : %fs\n",time);
			
			break;
		}
		
		printf("The %d th epoch, error: %f\n", counterEpochs, error);
		
		gettimeofday (&endwtime, NULL); // stop counting time
		time += (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("Total time        : %fs\n",time);
		
		counterEpochs++;
	}
	
	/* free all the variables used */
	
	free(activation[0]);
	for(int j=1 ; j<numberOfLayers ; j++){  
		free(z[j-1]);		
		free(activation[j]);	
		free(bias[j-1]);		
		free(weights[j-1]);
		free(delta[j-1]);
	}	
	
	free(errorArray);
	free(weights);
	free(z);	
	free(activation);
	free(bias);		
	free(delta);
	free(neuronsOfLayer);
	
	
	for(int i = 0; i < datasize; i++){	
		free(data[i].input);	
		free(data[i].teach);
	}	
	free(data);
	
	
}