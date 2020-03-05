#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include "pthread.h"

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
			  int sft, int lv);


struct radix_threads { // struct with which we pass multiples in the pthread_create() routine 
    unsigned long int *morton_codes;
    unsigned long int *sorted_morton_codes;
    unsigned int *permutation_vector;
    unsigned int *index;
    int *level_record;
    int N;
    int population_threshold;
    int sft;
    int lv;
};


/*
	Creation of parallel radix. 
	This code below is being parallelized through this routine that each thread will execute once it is created
	  

	   for(int i=0; i<MAXBINS; i++){
      
         truncated_radix_sort(&morton_codes[offset], 
			   &sorted_morton_codes[offset], 
			   &permutation_vector[offset], 
			   &index[offset], &level_record[offset], 
			   size, 
			   population_threshold,
			   sft-3, lv+1);
    }
*/



void  *parallel_radix(void * thread_data)
{
	/*  Passes the arguments of each thread 
		to the truncated_radix_sort() routine 
		so as to be executed recursively   */

	struct radix_threads *temp= (struct radix_threads *) thread_data; // typecasting the void struct to morton_threads structure
    unsigned long int *morton_codes = temp->morton_codes;
    unsigned long int *sorted_morton_codes = temp->sorted_morton_codes;
    unsigned int *permutation_vector = temp->permutation_vector;
    unsigned int *index = temp->index;
    int *level_record = temp->level_record;
    int N = temp->N;
    int population_threshold = temp->population_threshold;
    int sft = temp->sft;
    int lv = temp->lv;
	
	truncated_radix_sort(morton_codes,sorted_morton_codes,permutation_vector,
		index, level_record, N, population_threshold, sft, lv);

	 pthread_exit(NULL);
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

  int threadCheck;
  extern int numThreads;
  extern int enabledThreads;
  void *status;

  //call the recursive function in parallel if (numThreads-enabledThreads)>=8
  if((numThreads-enabledThreads)>=8){

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
	

		  pthread_t threads[MAXBINS]; //array of threads' identities
		  pthread_attr_t joinable; // declare a pthread attribure
		  pthread_attr_init(&joinable); // initialize the threads' attribute as joinable
		  pthread_attr_setdetachstate(&joinable, PTHREAD_CREATE_JOINABLE); //determines that all threads created are joinable

		  struct radix_threads thread_data[MAXBINS];

		  pthread_spinlock_t lock;
		  pthread_spin_init(&lock,PTHREAD_PROCESS_PRIVATE);
		  
		  // Call the function recursively to split the lower levels 
		   for(i=0; i<MAXBINS; i++){
			 
			 pthread_spin_lock(&lock);
			 enabledThreads++;
             pthread_spin_unlock(&lock);

			 thread_data[i].morton_codes = &morton_codes[offsets[i]];
			 thread_data[i].sorted_morton_codes = &sorted_morton_codes[offsets[i]];
			 thread_data[i].permutation_vector = &permutation_vector[offsets[i]];
			 thread_data[i].index = &index[offsets[i]];
             thread_data[i].level_record = &level_record[offsets[i]];
             thread_data[i].N = sizes[i];
             thread_data[i].population_threshold = population_threshold;
             thread_data[i].sft = sft-3;
             thread_data[i].lv = lv+1;

         
             threadCheck = pthread_create(&threads[i], &joinable, parallel_radix, (void *)&thread_data[i]);

		     if(threadCheck){
				  printf("Error: pthread_create returned code %d\n", threadCheck);
				  return;
			 }
			}

			pthread_attr_destroy(&joinable); // free attribute and wait for the other threads
			for (i=0;i<MAXBINS;i++){
				threadCheck = pthread_join(threads[i], &status);
				if (threadCheck){
					printf("Error; return code from pthread_join() is %d\n", threadCheck);
					exit(-1);
				}
			}

	 }
    
  }
  else
   //call the recursive function serially if (numThreads-enabledThreads)<8
  {
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

		  // Call the function recursively to split the lower levels 
		  for(int i=0; i<MAXBINS; i++){
			  int offset = (i>0) ? BinSizes[i-1] : 0;
			  int size = BinSizes[i] - offset;
      
			truncated_radix_sort(&morton_codes[offset], 
			   &sorted_morton_codes[offset], 
			   &permutation_vector[offset], 
			   &index[offset], &level_record[offset], 
			   size, 
			   population_threshold,
			   sft-3, lv+1);
		  }
    
      
       } 
	}
	
}
