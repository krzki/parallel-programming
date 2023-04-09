/*
    Try to exploit the fact that C use row-major order to store multidimensional matrix
*/

#define _GNU_SOURCE

#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "mpi.h"
#define N               2048        /* number of rows and columns in matrix */

MPI_Status status;

double a[N][N],b[N][N],c[N][N];

void transpose(int size, double[size][size]);


int main(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,rows,offset,i,j,k;

  struct timeval start, stop;
  double comm_start, comm_end;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  numworkers = numtasks-1;

  /*---------------------------- master ----------------------------*/
  if (taskid == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }
    
    gettimeofday(&start, 0);
    transpose(N, b);

    /* send matrix data to the worker tasks */
    if (numworkers > 0) {
        rows = N/numworkers;
    }
    else {
        rows = N;
    }

    offset = 0;
    comm_start = MPI_Wtime();
    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }
    comm_end = MPI_Wtime();

    /* wait for results from all worker tasks */
    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }

    if(numworkers == 0) {
        for(i = 0 ; i < rows; i++) {
            for (j = 0; j < N; j++) {
                c[i][j] = 0;
                for (k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[j][k];  
                }
            }
        }
    }

    transpose(N, b); // convert b back to its original matrix (B = (B^T)^T)

    gettimeofday(&stop, 0);

    // printf("Here is the result matrix:\n");
    // for (i=0; i<N; i++) {
    //   for (j=0; j<N; j++)
    //     printf("%6.2f   ", c[i][j]);
    //   printf ("\n");
    // }

    FILE *fp;
    char file_name[] = "MATMUL_ROW_MAJOR.txt";
    if ((fp = fopen(file_name, "a+")) == NULL) {
        fprintf(stdout,"Can't open \"%s\" file.\n", file_name);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "Matrix Size (N)= %d, Processor=%d\n", N, numworkers + 1);
    fprintf(fp,"\tTotal Time = %.6f\n",
         (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));
    
    fprintf(fp,"\tExecution Time = %.6f\n",
         (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6) - 7 * (comm_end - comm_start) / 4);
    fprintf(fp, "\tCommunication Time = %.6f\n\n", 7 * (comm_end - comm_start) / 4);   
    
    if (fclose(fp) != 0) {
        fprintf(stderr,"Error closing file\n");
    }


    // fprintf(stdout,"Execution Time = %.6f\n",
    //      (stop.tv_sec + stop.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6));
    
    // fprintf(stdout, "Communication Time = %.6f\n\n", 2 * (comm_end - comm_start));

  }

  /*---------------------------- worker----------------------------*/
  if (taskid > 0) {
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    
    // row-major matrix multiplication

    for(i = 0 ; i < rows; i++) {
        for (j = 0; j < N; j++) {
            c[i][j] = 0;
            for (k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[j][k];  
            }
        }
    }

    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}


void transpose(int size, double matrix[size][size]) {
    double temp;
    for(int i = 0; i < size - 1; i++) {
        for(int j = i; j < size; j++) {
            temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}