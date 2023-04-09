#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

void gen_matrix(int size, double matrix[size][size]);
void gen_vector(int size, double vector[size]);
int write_matrix(int size, double matrix[size][size]);
int write_vector(int size, double vector[size]);
void normalize_matrix(int size, double matrix[size][size]);

int main (int argc, char ** argv) {
    int size, my_rank, comm_sz;

    if(argc > 1) {
        size = atoi(argv[1]);
    }
    else {
        exit(EXIT_FAILURE);
    }

    double (* matrix)[size], *vector, (* local_matrix)[size];
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (size % comm_sz != 0) {
        fprintf(stderr, "Matrix size must be divisible by processor number\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_UNKNOWN);
    }

    matrix = (double (*)[size]) malloc (size * size * sizeof(double));
    local_matrix = (double (*)[size]) malloc ((size / comm_sz) * size * sizeof(double));
    if (my_rank == 0) {
        vector = (double *) malloc (size * sizeof(double));

        gen_matrix(size, matrix);
        gen_vector(size, vector);
    }
    MPI_Bcast(matrix, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_size = size / comm_sz;
    for(int i = 0; i < local_size; i++) {
        for(int j = 0; j < size; j++) {
            local_matrix[i][j] = 0;
            for(int k = 0; k < size; k++) {
                local_matrix[i][j] += matrix[i + my_rank * local_size][k] * matrix[j][k];
            }
        }
    }
    MPI_Gather(local_matrix, local_size * size, MPI_DOUBLE, matrix, local_size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Normalize matrix\n");
        normalize_matrix(size, matrix);

        printf("Start writing the matrix\n");
        write_matrix(size, matrix);

        printf("Start writing the vector\n");
        write_vector(size, vector);
        free(vector);    
    }
    free(matrix);
    free(local_matrix);
    MPI_Finalize();
    return 0;
}

void gen_matrix(int size, double matrix[size][size]) {
    srand(time(NULL));    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            matrix[i][j] = (double) rand() / RAND_MAX;
        
        }
    }

}


void gen_vector(int size, double vector[size]) {
    srand(time(NULL));

    for(int i = 0; i < size; i++) { 
        vector[i] = (double) rand() / RAND_MAX;
    }

}

void normalize_matrix(int size, double matrix[size][size]) {
    if (size == 0) {
        return;
    }
    double max = matrix[0][0];

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
            }
        }
    }

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            matrix[i][j] = matrix[i][j] / max;
        }
    }
}

int write_matrix(int size, double matrix[size][size]) {
    FILE *fp_matrix;
    char matrix_file_name[] = "MATRIX";
    if ((fp_matrix = fopen(matrix_file_name, "w+")) == NULL) {
        fprintf(stdout,"Can't open \"%s\" file.\n", matrix_file_name);
        return EXIT_FAILURE;
    }

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            fprintf(fp_matrix, "%.8f ", matrix[i][j]);
        }
        fprintf(fp_matrix, "\n");
    }    
    if (fclose(fp_matrix) != 0) {
        fprintf(stderr,"Error closing file\n");
        return EXIT_FAILURE;
    }
    else {
        printf("Done writing matrix with size %dx%d.\n", size, size);
        return EXIT_SUCCESS;
    }
}

int write_vector(int size, double vector[size]) {
    FILE *fp_vector;
    char vector_file_name[] = "VECTOR";
    if ((fp_vector = fopen(vector_file_name, "w+")) == NULL) {
        fprintf(stdout,"Can't open \"%s\" file.\n", vector_file_name);
        return EXIT_FAILURE;
    }

    for(int i = 0; i < size; i++) {
        fprintf(fp_vector, "%.8f ", vector[i]);
    }    
    if (fclose(fp_vector) != 0) {
        fprintf(stderr,"Error closing file\n");
        return EXIT_FAILURE;
    }
    else {
        printf("Done writing matrix with size %d.\n", size);
        return EXIT_SUCCESS;
    }
}