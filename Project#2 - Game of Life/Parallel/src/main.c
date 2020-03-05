/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>
#include <game-of-life.h>
#include "sys/time.h"


int main (int argc, char *argv[]) {
	
  struct timeval startwtime, endwtime; // Time counting variables
  gettimeofday (&startwtime, NULL); 
  
  extern int numThreads; // Variable to declare the number of threads to be used in each parallel region
  int numtasks, rank,i ; // Variable to declare the number of processes according to the input (M,N)

  if (argc != 7) { // Check if the command line arguments are correct 
    printf("Usage: %s N thres disp\n"
	   "where\n"
	   "  M     : size of table (M x N)\n"
	   "  N     : size of table (M x N)\n"
	   "  thres : propability of alive cell\n"
       "  t     : number of generations\n"
	   "  disp  : {1: display output, 0: hide output}\n"
       "numThreads : number of threads.\n" , argv[0]);
    return (3);
  }
  
  // Input command line arguments
  int M = atoi(argv[1]);        // Array size
  int N = atoi(argv[2]);		// Array size
  double thres = atof(argv[3]); // Propability of life cell
  int t = atoi(argv[4]);        // Number of generations 
  int disp = atoi(argv[5]);     // Display output?
  numThreads = atoi(argv[6]);   // number of threads
  printf("Size %dx%d with propability: %0.1lf%%\n", M, N, thres*100);

  
  MPI_Init(&argc,&argv); // Initialize MPI
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks); // Determines the number of processes in the group of comm communicator
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); // Determines the rank of the calling process in the communicator
	
  printf("rank %d : \n",rank);

  if (M==40000 && N==40000){
	// Assign the work in one process
	
	int *board=malloc(sizeof(int)*N*N);
	if (board == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	/* second pointer for updated result */
	int *newboard=malloc(sizeof(int)*N*N);
	if (newboard == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	generate_table (board, M, N, thres);
	printf("Board generated\n");

	 /* play game of life t times */
	for (i=0; i<t; i++) {
		play (board, newboard, M, N);   
	}

	free(board);
	free(newboard);
	
  } else if (M==80000 && N==40000){   
	// Devide the initial 80000*40000 board into two 40000*40000 boards and assign the work in two processes	
	
	MPI_Request reqs[4]; // Required variable for non-blocking calls
    MPI_Status stats[4]; // Required variable for Waitall routine
	
	int *board=malloc(sizeof(int)*N*N);
	if (board == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	/* second pointer for updated result */
	int *newboard=malloc(sizeof(int)*N*N);
	if (newboard == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}

	int *first_row=malloc(sizeof(int)*N); // Contains the elements of the first row of another processes' board
	if (first_row == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	int *last_row=malloc(sizeof(int)*N); // Contains the elements of the last row of another processes' board
	if (last_row == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	generate_table (board, N, N, thres);
	printf("Board generated\n");

	/* play game of life t times */
	for (i=0; i<t; i++) {
		
		// Communication between two processes
		if (rank==0){
			MPI_Isend(board, N, MPI_INT, 1, 0, MPI_COMM_WORLD, &reqs[0]);
			MPI_Isend(&Board(N-1,0), N, MPI_INT, 1, 1, MPI_COMM_WORLD, &reqs[1]);

			MPI_Irecv(first_row, N, MPI_INT, 1, 2, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(last_row, N, MPI_INT, 1, 3, MPI_COMM_WORLD, &reqs[3]);

		}else if (rank==1){
			MPI_Isend(board, N, MPI_INT, 0, 2, MPI_COMM_WORLD, &reqs[0]);
			MPI_Isend(&Board(N-1,0), N, MPI_INT, 0, 3, MPI_COMM_WORLD, &reqs[1]);

			MPI_Irecv(first_row, N, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(last_row, N, MPI_INT, 0, 1, MPI_COMM_WORLD, &reqs[3]);	
		}
		
		play_processes (board, newboard, N, N, first_row, last_row, reqs, stats);    
	}
	
	free(board);
	free(newboard);
	free(first_row);
	free(last_row);

  } else if (M==80000 && N==80000){
	// Devide the initial 80000*80000 board into four 40000*40000 boards and assign the work in four processes
	  
	int *board=malloc(sizeof(int)*N*N/4);
	if (board == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	/* second pointer for updated result */
	int *newboard=malloc(sizeof(int)*N*N/4);
	if (newboard == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	generate_table (board, M/4, N, thres);
	printf("Board generated\n");

    MPI_Request reqs[4];
    MPI_Status stats[4];

	int *first_row= malloc(sizeof(int)*N); // Contains the elements of the first row of another processes' board
	if (first_row == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	int *last_row= malloc(sizeof(int)*N); // Contains the elements of the last row of another processes' board
	if (last_row == NULL){
		printf("\n ERROR: Memory allocation did not complete successfully!\n");
		return (1);
	}
	
	/* play game of life t times */
	 
	 for (i=0; i<t; i++) {
		
		// Communication between four processes
		if (rank==0){
			
			MPI_Isend(board, N, MPI_INT, 3, 0, MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(first_row, N, MPI_INT, 1, 1, MPI_COMM_WORLD, &reqs[1]);

			MPI_Isend(&Board(N/4-1,0), N, MPI_INT, 1, 2, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(last_row, N, MPI_INT, 3, 3, MPI_COMM_WORLD, &reqs[3]);
			
		}else if (rank==1){
	
			MPI_Isend(board, N, MPI_INT, 0, 1, MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(first_row, N, MPI_INT, 2, 4, MPI_COMM_WORLD, &reqs[1]);

			MPI_Isend(&Board(N/4-1,0), N, MPI_INT, 2, 5, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(last_row, N, MPI_INT, 0, 2, MPI_COMM_WORLD, &reqs[3]);
			
		}else if (rank==2){
			
			MPI_Isend(board, N, MPI_INT, 1, 4, MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(first_row, N, MPI_INT, 3, 6, MPI_COMM_WORLD, &reqs[1]);

			MPI_Isend(&Board(N/4-1,0), N, MPI_INT, 3, 7, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(first_row, N, MPI_INT, 1, 5, MPI_COMM_WORLD, &reqs[3]);
			
		}else{
			
			MPI_Isend(board, N, MPI_INT, 2, 6, MPI_COMM_WORLD, &reqs[0]);
			MPI_Irecv(first_row, N, MPI_INT, 0, 0, MPI_COMM_WORLD, &reqs[1]);

			MPI_Isend(&Board(N/4-1,0), N, MPI_INT, 0, 3, MPI_COMM_WORLD, &reqs[2]);
			MPI_Irecv(last_row, N, MPI_INT, 2, 7, MPI_COMM_WORLD, &reqs[3]);
			
		}

		play_processes (board, newboard, M/4, N, first_row, last_row, reqs, stats);    
	
	}

	free(board);
	free(newboard);
	free(first_row);
	free(last_row);

	}else{
		printf("Wrong input M or N\n");
		exit(0);
	}
  
	printf("Game finished after %d generations.\n", t);

	gettimeofday (&endwtime, NULL);
	double time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    
    printf("Time to compute the process with rank %d : %fs\n",rank, time);

	
	MPI_Finalize();
  	
}
