/*
 * CUDA
 */
 
#include <math.h>
#include <stdio.h>
#include <cuda.h>

// Array access macro
#define Image(x,y) J[(x)*n + (y)]

__global__ void CudaKernel(float *J, int m, int n, int neib, float filtSigma, float const* const Gauss) {
	__shared__ float sharedGauss[49];  
	__shared__ float sharedJ[8][8][49];  
	
	int index= threadIdx.x;  //unique index for each thread in a block
	
	/*read array Gauss from global memory to shared memory*/
	while (index>=0 && index<((2*neib+1)*(2*neib+1))){
		sharedGauss[index] = Gauss[index];
		index += blockDim.x;
	}
	
	//pixel (pi,pj) is the one that we want to denoise
	int pi = blockIdx.y * blockDim.y + threadIdx.y;  //unique index for each thread with blockDim.y threads per block
	int pj = blockIdx.x * blockDim.x + threadIdx.x;  //unique index for each thread with blockDim.x threads per block
	
	/*create a 3D array in shared memory so as each thread (pi,pj) does not search 
	for its neighbors in global memory each time*/
	int t = 0;
	if (pi>=neib && pi<=m-1-neib && pj>=neib && pj<=n-1-neib){		
		for (int i=pi-neib; i<=(pi+neib); i++){
			for (int j=pj-neib; j<=(pj+neib); j++){
				sharedJ[threadIdx.y][threadIdx.x][t] = Image(i,j);
				t++;
			}
		}
	}	
	
	__syncthreads();  //synchronize all threads within a block
		
	int qi, qj;  //indexes for all pixels of the image that will be used for denoising pixel (pi,pj)
	float sum, e, denoisedP = 0.0, Z = 0.0;

	if (pi>=neib && pi<=m-1-neib && pj>=neib && pj<=n-1-neib){		
		
		/*first pass — compute NLM-weights*/
		for (qi=neib; qi<=m-neib; qi++){
			for (qj=neib; qj<=n-neib; qj++){
				sum = 0.0;
				t = 0;
				for(int k=-neib; k<=neib; k++){
					for(int l=-neib; l<=neib; l++){
						
						//compute distance between two patches with center pixels (pi,pj) and (qi,qj)					
						sum += (((sharedJ[threadIdx.y][threadIdx.x][t]-Image(qi+k,qj+l))*(sharedJ[threadIdx.y][threadIdx.x][t]-Image(qi+k,qj+l)))*sharedGauss[t]);
						t++;				
					}
				}
				e = exp(-sum/filtSigma);  //compute unnormalized weight w(x,y)
				Z += e;  //compute sum of unnormalized weights w(x,y)
			}					
		}
		
		/*second pass — compute denoised pixel*/
		for (qi=neib; qi<=m-neib; qi++){
			for (qj=neib; qj<=m-neib; qj++){
				sum = 0.0;
				t = 0;
				for(int k=-neib; k<=neib; k++){
					for(int l=-neib; l<=neib; l++){
										
						//compute distance between two patches with center pixels (pi,pj) and (qi,qj)
						sum += (((sharedJ[threadIdx.y][threadIdx.x][t]-Image(qi+k,qj+l))*(sharedJ[threadIdx.y][threadIdx.x][t]-Image(qi+k,qj+l)))*sharedGauss[t]);
						t++;

					}
				}
				e = exp(-sum/filtSigma);  //compute unnormalized weight w(x,y)
				denoisedP += e/Z*Image(qi,qj);  //compute sum of normalized weights w(x,y)
			}					
		}
		Image(pi,pj) = denoisedP;	
	}	
}