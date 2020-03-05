/*
 *	Utils
 */
 
struct DATA{
	int *input;
	int *teach;
};	
 
/* Random Generators */

void generate(struct DATA *data, int datasize, int M, int N);

void *generate_parallel(void *thread_data);

void generate_data(int numberOfThreads, double **weights, double **bias, int numberOfLayers, int *neuronsOfLayer);

void *generate_data_parallel(void *thread_data);


/* Feed forward */

void launchKernelF(int numberOfLayers, int *neuronsOfLayer, double **weights, double **activation, double **bias, double **z);

/* Output error calculation */

void launchKernelE(double *errorArray, int numberOfLayers, int *neuronsOfLayer, int *teach, double **delta, double **activation, double **z);

/* Backpropagation */


void launchKernelB(int numberOfLayers, int *neuronsOfLayer, double **weights, double **delta, double **z);


/* Updateweights */


void launchKernelU(double eta, int datasize,double **weights, double **delta, double **activation, int numberOfLayers, int *neuronsOfLayer, double **bias);
