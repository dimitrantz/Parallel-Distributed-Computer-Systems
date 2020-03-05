#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <game-of-life.h>

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N) {
  i += a;
  while (i < 0) i += N;
  while (i >= N) i -= N;
  return i;
}

/* add to a height index, wrapping around */

int yadd (int i, int a, int N) {
  i += a;
  while (i < 0) i += N;
  while (i >= N) i -= N;
  return i;
}

/* return the number of on cells adjacent to the i,j cell */


// adjacent to for only one process
int adjacent_to (int *board, int i, int j,int M, int N) {
	
	int k, l, count;
	count = 0;
	
	/* go around the cell */
	for (k=-1; k<=1; k++){
		for (l=-1; l<=1; l++){
			/* only count if at least one of k,l isn't zero */
			if (k || l){
				if (Board(xadd(i,k,M),yadd(j,l,N))) count++;				
			}
		}
	}
	return count;
}

// adjacent to for only one process
int adjacent_to_less (int *board, int i, int j,int M, int N) {
	
	int k, l, count;
	count = 0;
	
	/* go around the cell */
	for (k=-1; k<=1; k++){
		for (l=-1; l<=1; l++){
			/* only count if at least one of k,l isn't zero */
			if (k || l){
				if (Board(i+k,yadd(j,l,N))) count++; // No need to call xadd() function
			}
		}
	}
	return count;
}

int adjacent_to_row (int *board, int i, int j,int M, int N, int *row, int start, int end) {
	
	int k, l, count;
	count = 0;

	    for (l=-1;l<=1;l++){
			if(row[yadd(j,l,N)]) count++; // Check the data sent by another process and calculate the neighbours
			for (k=start; k<=end; k++){
				/* only count if at least one of k,l isn't zero */
				if (k || l){
					if (Board(i+k,yadd(j,l,N))) count++; // No need to call xadd() function
				}
			}
		}


  return count;

}