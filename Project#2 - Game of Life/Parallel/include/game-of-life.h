/* #ifndef UTILS_H_   /\* Include guard *\/ */
/* #define UTILS_H_ */

#define Board(x,y) board[(x)*N + (y)]
#define NewBoard(x,y) newboard[(x)*N + (y)]

/* set everthing to zero */

void initialize_board (int *board, int N);

/* add to a width index, wrapping around like a cylinder */

int xadd (int i, int a, int N);

/* add to a height index, wrapping around */

int yadd (int i, int a, int N);

/* return the number of on cells adjacent to the i,j cell */

int adjacent_to (int *board, int i, int j,int M, int N);
int adjacent_to_less (int *board, int i, int j,int M, int N);
int adjacent_to_row (int *board, int i, int j,int M, int N, int *row, int start, int end);

/* play the game through one generation */
void play (int *board, int *newboard, int M, int N);
void play_processes (int *board, int *newboard, int M, int N, int *first_row, int *last_row, MPI_Request *reqs, MPI_Status *stats);

/* print the life board */

void print (int *board, int N);

/* generate random table */

void generate_table (int *board, int M, int N, float threshold);

/* display the table with delay and clear console */

void display_table(int *board, int N);

/* #endif // FOO_H_ */

 int numThreads;
 