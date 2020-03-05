#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <game-of-life.h>

/* set everthing to zero */

void initialize_board (int *board, int N) {
  int   i, j;
  
  for (i=0; i<N; i++)
    for (j=0; j<N; j++) 
      Board(i,j) = 0;
}

/* generate random table */

void generate_table (int *board, int N, float threshold) {

  int   i, j;
  int counter = 0;

  srand(time(NULL));

  for (j=0; j<N; j++) {

    for (i=0; i<N; i++) {
      Board(i,j) = ( (float)rand() / (float)RAND_MAX ) < threshold;
      counter += Board(i,j);
    }
  }
}

