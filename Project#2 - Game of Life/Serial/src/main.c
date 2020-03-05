/*
 * Game of Life implementation based on
 * http://www.cs.utexas.edu/users/djimenez/utsa/cs1713-3/c/life.txt
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

int main (int argc, char *argv[]) {
  int   *board, *newboard, i;

  if (argc != 5) { // Check if the command line arguments are correct 
    printf("Usage: %s N thres disp\n"
	   "where\n"
	   "  N     : size of table (N x N)\n"
	   "  thres : propability of alive cell\n"
           "  t     : number of generations\n"
	   "  disp  : {1: display output, 0: hide output}\n"
           , argv[0]);
    return (1);
  }
  
  // Input command line arguments
  int N = atoi(argv[1]);        // Array size
  double thres = atof(argv[2]); // Propability of life cell
  int t = atoi(argv[3]);        // Number of generations 
  int disp = atoi(argv[4]);     // Display output?
  printf("Size %dx%d with propability: %0.1f%%\n", N, N, thres*100);

  board = NULL;
  newboard = NULL;
  
  board = (int *)malloc(N*N*sizeof(int));

  if (board == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }

  /* second pointer for updated result */
  newboard = (int *)malloc(N*N*sizeof(int));

  if (newboard == NULL){
    printf("\nERROR: Memory allocation did not complete successfully!\n");
    return (1);
  }

  initialize_board (board, N);
  printf("Board initialized\n");
  generate_table (board, N, thres);
  printf("Board generated\n");

  /* play game of life 100 times */

  for (i=0; i<t; i++) {
    if (disp) display_table (board, N);
    play (board, newboard, N);    
  }
  printf("Game finished after %d generations.\n", t);
}
