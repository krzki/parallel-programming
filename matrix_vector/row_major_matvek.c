#define _GNU_SOURCE

#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#define N               2048        /* number of rows and columns in matrix */

double a[N][N], b[N], c[N];

MPI_Status status;

int main(int argc, char ** argv) {

    int numtasks, taskid, numworkers, rows, offset;

    double start, end, comm_start, comm_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    /*---------------------------- master ------------------*/ 
    if (taskid == 0 ) {
        srand(time(NULL));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = (double) rand() / RAND_MAX;
            }
        }
        for (int i = 0; i < N; i++) {
            b[i] = (double) rand() /  RAND_MAX;
        }

        start = MPI_Wtime();
        
        if (numworkers > 0) {
            rows = N / numworkers;
        } else {
            rows = N;
        }

        offset = 0;
        comm_start = MPI_Wtime();
	    int send_rows = rows;
        for (int dest = 1; dest <= numworkers; dest++) {
		
            if (dest == numworkers) {
                send_rows = N - offset;
            }

            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&send_rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(a[offset], send_rows * N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        comm_end = MPI_Wtime();

        for (int source = 1; source <= numworkers; source++) {
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset], rows, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        /*--------------- Sequential ---------------*/
        if(numworkers == 0) {
           for(int i = 0 ; i < N; i++) {
                c[i] = 0;
                for (int j = 0; j < N; j++) {
                    c[i] += a[i][j] * b[j];
                }
            }
        }

        end = MPI_Wtime();


        // save report to file
        FILE *fp;
        char * file_name = "MATVEK_ROW_MAJOR";

        if ((fp = fopen(file_name, "a+")) == NULL) {
            fprintf(stdout,"Can't open \"%s\" file.\n", file_name);
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "Matrix Size (N)= %d, Processor=%d\n", N, numworkers + 1);
        fprintf(fp, "Total Time: %.6f\n", end - start);
        fprintf(fp, "Execution Time: %.6f\n", end - start - (1.0 / N + 1) * (comm_end - comm_start));
        fprintf(fp, "Communication Time: %.6f\n\n", 7 * (comm_end - comm_start) / 4);   
        
        if (fclose(fp) != 0) {
            free(file_name);
            fprintf(stderr,"Error closing file\n");
        }
        free(file_name);
    }

    /*---------------------------- worker----------------------------*/
    if (taskid > 0) {
        int source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(b, N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
 
        // row-major matmul

        for(int i = 0 ; i < rows; i++) {
            c[i] = 0;
            for (int j = 0; j < N; j++) {
                c[i] += a[i][j] * b[j];
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(c, rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
