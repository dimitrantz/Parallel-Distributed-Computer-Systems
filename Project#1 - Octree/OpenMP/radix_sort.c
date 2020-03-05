#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "math.h"
#include <omp.h>


#define MAXBINS 8


inline void swap_long(unsigned long int **x, unsigned long int **y){

  unsigned long int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

inline void swap(unsigned int **x, unsigned int **y){

  unsigned int *tmp;
  tmp = x[0];
  x[0] = y[0];
  y[0] = tmp;

}

void truncated_radix_sort(unsigned long int *morton_codes, 
			  unsigned long int *sorted_morton_codes, 
			  unsigned int *permutation_vector,
			  unsigned int *index,
			  unsigned int *level_record,
			  int N, 
			  int population_threshold,
			  int sft, int lv){


  int BinSizes[MAXBINS] = {0};
  int BinCursor[MAXBINS] = {0};

  unsigned int *tmp_ptr;
  unsigned long int *tmp_code;

  extern int numThreads;
  extern int enabledThreads;
  
  if(N<=0){

    return;
  }
  else if(N<=population_threshold || sft < 0) { // Base case. The node is a leaf

    level_record[0] = lv; // record the level of the node
    memcpy(permutation_vector, index, N*sizeof(unsigned int)); // Copy the pernutation vector
    memcpy(sorted_morton_codes, morton_codes, N*sizeof(unsigned long int)); // Copy the Morton codes 

    return;
  }
  else{

    level_record[0] = lv;
    // Find which child each point belongs to 
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      BinSizes[ii]++;
    }

    // scan prefix (must change this code)  
    int offset = 0;
    for(int i=0; i<MAXBINS; i++){
      int ss = BinSizes[i];
      BinCursor[i] = offset;
      offset += ss;
      BinSizes[i] = offset;
    }
   
    for(int j=0; j<N; j++){
      unsigned int ii = (morton_codes[j]>>sft) & 0x07;
      permutation_vector[BinCursor[ii]] = index[j];
      sorted_morton_codes[BinCursor[ii]] = morton_codes[j];
      BinCursor[ii]++;
    }


    //swap the index pointers  
    swap(&index, &permutation_vector);

    //swap the code pointers 
    swap_long(&morton_codes, &sorted_morton_codes);
 
    //Call the function recursively to split the lower levels

	/*  
		Store the context of these variables in two new matrixes 
		to make sure threads get the right arguments 

		int offset = (i>0) ? BinSizes[i-1] : 0;
        int size = BinSizes[i] - offset;
	*/

	int i;
    int sizes[MAXBINS];
	int offsets[MAXBINS];

	sizes[0]=BinSizes[0];
	offsets[0]=0;

    for (i=1; i<MAXBINS; i++) {
        sizes[i] = BinSizes[i]-BinSizes[i-1];
		offsets[i] = BinSizes[i-1];
    }

	#pragma omp flush(enabledThreads) // specifies that all threads have the same view of memory for this variable
    omp_set_nested(enabledThreads<=numThreads); // enable nested parallelism if possible

	#pragma omp flush(enabledThreads)
	// set the number of threads that can be created for the parallel region below
	omp_set_num_threads((numThreads>=enabledThreads) ? numThreads-enabledThreads: 0);


	#pragma omp parallel num_threads(omp_get_num_threads()) private(i)
	{
	
		#pragma omp for schedule(static) nowait
		  for(i=0; i<MAXBINS; i++){
	
			  if (omp_get_nested()){// if nested parallelism is enabled, increases the number of threads working
                  
                    #pragma omp atomic // specifies that this variable will be updated each time by one thread  
                    enabledThreads ++; 
                    #pragma omp flush(enabledThreads)
			  }
      
			  truncated_radix_sort(&morton_codes[offsets[i]], 
				   &sorted_morton_codes[offsets[i]], 
				   &permutation_vector[offsets[i]], 
				   &index[offsets[i]], &level_record[offsets[i]], 
				   sizes[i], 
				   population_threshold,
				   sft-3, lv+1);
	  
		      if (omp_get_nested()){ // if nested parallelism was enabled, a thread is about to terminate
                   
                    #pragma omp atomic
                    enabledThreads--; 
                    #pragma omp flush(enabledThreads)
		      }
	      
        } 
    }
 }
}
