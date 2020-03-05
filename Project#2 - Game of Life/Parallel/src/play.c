#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include "mpi.h"
#include <game-of-life.h>



// play for one process
void play (int *board, int *newboard, int M, int N) {  
	//omp_set_nested(1);
	//printf("helloo from play \n");
	/*
    (copied this from some web page, hence the English spellings...)

    1.STASIS : If, for a given cell, the number of on neighbours is 
    exactly two, the cell maintains its status quo into the next 
    generation. If the cell is on, it stays on, if it is off, it stays off.

    2.GROWTH : If the number of on neighbours is exactly three, the cell 
    will be on in the next generation. This is regardless of the cell's
    current state.

    3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will 
    be off in the next generation.
  */
  
	int i;
	extern int numThreads;
	
	#pragma omp parallel num_threads(numThreads) private(i)
	{
		#pragma omp for schedule(static) nowait // Schedule is defined as static due to the equivalent work of each thread
		for (i=0; i<N; i++){
		   for (int j=0; j<N; j++) {
			  int a = adjacent_to (board, i, j, M, N); // Calculates the alive neighbours for each cell of the board
			  if (a == 2) NewBoard(i,j) = Board(i,j);
			  if (a == 3) NewBoard(i,j) = 1;
			  if (a < 2) NewBoard(i,j) = 0;
			  if (a > 3) NewBoard(i,j) = 0;
			}
		
		}
	 
	}
  /* copy the new board back into the old board by swapping the pointers*/
	
	int *temp;
    temp = board;
	board = newboard;
	newboard = temp;

}

// play for two processes
void play_processes (int *board, int *newboard, int M, int N, int *first_row, int *last_row, MPI_Request *reqs, MPI_Status *stats) { 

	//printf("helloo from play_two \n");
  /*
    (copied this from some web page, hence the English spellings...)

    1.STASIS : If, for a given cell, the number of on neighbours is 
    exactly two, the cell maintains its status quo into the next 
    generation. If the cell is on, it stays on, if it is off, it stays off.

    2.GROWTH : If the number of on neighbours is exactly three, the cell 
    will be on in the next generation. This is regardless of the cell's
    current state.

    3.DEATH : If the number of on neighbours is 0, 1, 4-8, the cell will 
    be off in the next generation.
  */
  
  int i,k, flag;
  extern int numThreads;

  /* for each cell, apply the rules of Life */
	#pragma omp parallel num_threads(numThreads) private(i)
	{	
	   #pragma omp for schedule(static) nowait // Schedule is defined as static due to the equivalent work of each thread
		for (i=1; i<M-1; i++){
		    for (int j=0; j<N; j++) {
			  int a = adjacent_to_less (board, i, j, M, N); // Calculates the alive neighbours for each cell except those in the first and the last row of the board 
			  if (a == 2) NewBoard(i,j) = Board(i,j);
			  if (a == 3) NewBoard(i,j) = 1;
			  if (a < 2) NewBoard(i,j) = 0;
			  if (a > 3) NewBoard(i,j) = 0;
			}
		}
	}
	
	MPI_Testall(4, reqs, &flag, stats); // Check the state of the communication and wait if it is not completed
	if (!flag) MPI_Waitall(4, reqs, stats); 
	
	#pragma omp parallel num_threads(numThreads) private(k)
	{
		#pragma omp for schedule(static) nowait // Schedule is defined as static due to the equivalent work of each thread
		for (k=0; k<N; k++) {
			// i=0;
			int b = adjacent_to_row (board, 0, k, M, N, last_row,0,1); // Calculates the alive neighbours for each cell of the first row of the board
			if (b == 2) NewBoard(0,k) = Board(0,k);
			if (b == 3) NewBoard(0,k) = 1;
			if (b < 2) NewBoard(0,k) = 0;
			if (b > 3) NewBoard(0,k) = 0;
		
			// i=M-1;
			int c = adjacent_to_row(board, M-1, k, M, N, first_row,-1,0); // Calculates the alive neighbours for each cell of the last row of the board
			if (c == 2) NewBoard(M-1,k) = Board(M-1,k);
			if (c == 3) NewBoard(M-1,k) = 1;
			if (c < 2) NewBoard(M-1,k) = 0;
			if (c > 3) NewBoard(M-1,k) = 0;
		}
	}
  
  /* copy the new board back into the old board by swapping the pointers*/
  
	int *temp;
    temp = board;
	board = newboard;
	newboard = temp;

}

  