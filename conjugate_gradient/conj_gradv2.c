/*
    Paralel Conjugate Gradient (MPI version)
    MPI     : Open MPI 4.1.5

    Reference:

    Conjugate Gradient Algorithm
    Rauber, T., & Rünger, G. (2010). Parallel Programming: For Multicore and Cluster Systems.
        Springer Science & Business Media.
    
    Symmetric positif-definit matrix generator
    https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab 
*/

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define MAX_ITER 1000
#define TOLERANCE 10e-8

void seq_conj_grad(int size, const double A[size][size], const double b[size], double x[size]);
double dot_product(int size, const double vec_1[size], const double vec_2[size]);
void vector_add(int size, const double vec_1[size], const double vec_2[size], double res[size]);
void vector_sub(int size, const double vec_1[size], const double vec_2[size], double res[size]);
void constant_multiply_vector(int size, const double vec[size], double multiplier, double res[size]);
void vector_copy(int size, const double source[size], double dest[size]);
void print_matrix(int row, int col, const double matrix[row][col]);
void axpy(int size, double a, const double x[size], double y[size]);
void axpy_save_z(int size, double a, const double x[size], const double y[size], double z[size]);
void gen_vector(int size, double vector[size]);
void gen_symmetric_matrix(int size, double matrix[size][size]);
void matrix_vector_multiply(int row, int col, const double A[row][col], const double vec[col], double res[row]);
double relative_error(int size, const double A[size][size], const double x[size], const double b[size]);
int read_vector(int size, double vector[size]);
int read_matrix(int size, double matrix[size][size]);


int main(int argc, char **argv) {

    int comm_sz, my_rank, size;
    MPI_Status status;
    
    int *mat_strip, *vec_strip, *mat_displ, *vec_displ;
    
    double (* local_A)[size], *local_x, *local_b;
    double *global_d, global_g_dot_g, global_d_dot_w;
    
    double (* global_A)[size], *global_b, *global_init_x;
    double * global_x, *seq_x;
    
    double local_g_dot_g, local_d_dot_w, global_alpha, global_beta;
    double *tmp, *local_w, *local_g, *local_d, tmp_scalar; 
    
    double seq_start, seq_end, par_start, par_end, comm_total, comm_tmp;
    comm_total = 0;

    size = 16;
    if (argc > 1) {
        size = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    mat_strip = (int *) malloc (comm_sz * sizeof(int));
    vec_strip = (int *) malloc (comm_sz * sizeof(int));
    mat_displ = (int *) malloc (comm_sz * sizeof(int));
    vec_displ = (int *) malloc (comm_sz * sizeof(int));
    
    int unallocated_vector_size = size;
    int largest_vector_part = ((size + comm_sz - 1) / comm_sz); 

    for(int i = 0; i < comm_sz; i++) {
        if (unallocated_vector_size >= largest_vector_part) {
            vec_strip[i] = largest_vector_part;    
        }
        else {
            vec_strip[i] = unallocated_vector_size;              
        }

        mat_strip[i] = vec_strip[i] * size;
        
        vec_displ[i] = vec_displ[i - 1] + vec_strip[i - 1];
        mat_displ[i] = mat_displ[i - 1] + mat_strip[i - 1];  
        
        unallocated_vector_size -= vec_strip[i];
    }


    local_A = (double (*)[size]) malloc (mat_strip[my_rank] * sizeof(double));
    local_b = (double *) malloc (vec_strip[my_rank] * sizeof(double));
    global_x = (double *) malloc (size * sizeof(double));

    if (my_rank == 0) {
        global_A = (double (*)[size]) malloc (size * size * sizeof(double));
        global_b = (double *) malloc (size * sizeof(double));

        if (read_matrix(size, global_A) == EXIT_FAILURE) {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_FILE);
        }
        if (read_vector(size, global_b) == EXIT_FAILURE) {
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_FILE);
        }
        gen_vector(size, global_x);

        seq_x = (double *) malloc (size * sizeof(double));
        vector_copy(size, global_x, seq_x);

        /* ------------------- sequential start -------------------*/
        seq_start = MPI_Wtime();
        seq_conj_grad(size, global_A, global_b, seq_x);
        seq_end = MPI_Wtime();
        /* ------------------- sequential end   -------------------*/

        par_start = MPI_Wtime();
        comm_tmp = MPI_Wtime();
        MPI_Scatterv(
            global_A, mat_strip, mat_displ, MPI_DOUBLE,
            local_A, mat_strip[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD
        );
        
        MPI_Scatterv(
            global_b, vec_strip, vec_displ, MPI_DOUBLE,
            local_b, vec_strip[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD
        );
        comm_total += MPI_Wtime() - comm_tmp;
    } 

    else {

        MPI_Scatterv(
            NULL, NULL, NULL, NULL,
            local_A, mat_strip[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD
        );
        
        MPI_Scatterv(
            NULL, NULL, NULL, NULL,
            local_b, vec_strip[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD
        );
    }

    comm_tmp = MPI_Wtime();
    MPI_Bcast(global_x, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_total += MPI_Wtime() - comm_tmp;

    local_x = &global_x[vec_displ[my_rank]];

    int l_size = vec_strip[my_rank];

    local_w = (double *) malloc (vec_strip[my_rank] * sizeof(double));
    local_g = (double *) malloc (vec_strip[my_rank] * sizeof(double));
    local_d = (double *) malloc (vec_strip[my_rank] * sizeof(double));
    tmp = (double *) malloc (vec_strip[my_rank] * sizeof(double));
    global_d = (double *) malloc (size * sizeof(double));

    matrix_vector_multiply(l_size, size, local_A, global_x, tmp);
    vector_sub(l_size, local_b, tmp, local_d);
    vector_copy(l_size, local_d, local_g);

    comm_tmp = MPI_Wtime();
    MPI_Allgatherv(local_d, l_size, MPI_DOUBLE, global_d, vec_strip, vec_displ, MPI_DOUBLE, MPI_COMM_WORLD);
    comm_total += MPI_Wtime() - comm_tmp;

    global_g_dot_g = dot_product(size, global_d, global_d);


    int iter = 0;

    while (iter < MAX_ITER && pow(global_g_dot_g, 0.5) > TOLERANCE) {

        matrix_vector_multiply(l_size, size, local_A, global_d, local_w);

        local_d_dot_w = dot_product(l_size, local_d, local_w);

        comm_tmp = MPI_Wtime();
        MPI_Allreduce(&local_d_dot_w, &global_d_dot_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm_total += MPI_Wtime() - comm_tmp;
        
        global_alpha = global_g_dot_g / global_d_dot_w;
        
        axpy(l_size, global_alpha, local_d, local_x);
        axpy(l_size, -1 * global_alpha, local_w, local_g);

        tmp_scalar = global_g_dot_g;
        local_g_dot_g = dot_product(l_size, local_g, local_g);

        comm_tmp = MPI_Wtime();
        MPI_Allreduce(&local_g_dot_g, &global_g_dot_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm_total += MPI_Wtime() - comm_tmp;
        
        global_beta = global_g_dot_g / tmp_scalar;
        
        axpy_save_z(l_size, global_beta, local_d, local_g, local_d);

        comm_tmp = MPI_Wtime();
        MPI_Allgatherv(local_d, l_size, MPI_DOUBLE, global_d, vec_strip, vec_displ, MPI_DOUBLE, MPI_COMM_WORLD);
        comm_total += MPI_Wtime() - comm_tmp;

        iter++;
    }

    comm_tmp = MPI_Wtime();
    MPI_Gatherv(local_x, l_size, MPI_DOUBLE, global_x, vec_strip, vec_displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_total += MPI_Wtime() - comm_tmp;

    par_end = MPI_Wtime();
    
    // report and cleanup
    if(my_rank == 0) {
        printf("Matrix size: %d, processor: %d\n\n", size, comm_sz);

        printf("Sequential Relative Error : %.6f\n", relative_error(size, global_A, seq_x, global_b));
        printf("Paralel Relative Error    : %.6f\n\n", relative_error(size, global_A, global_x, global_b));

        printf("Sequential Total Time     : %.6f\n", seq_end - seq_start);
        printf("Paralel Total Time        : %.6f\n", par_end - par_start);
        printf("Paralel Communication Time: %.6f\n", comm_total);
        printf("Paralel Execution Time    : %.6f\n", par_end - par_start - comm_total);
        
        printf("=============================================\n\n");

        free(global_A);
        free(global_b);
        free(seq_x);
    }

    local_x = NULL;
    free(local_w);
    free(local_g);
    free(local_d);
    free(tmp);
    free(global_d);

    free(local_A);
    free(local_b);
    free(global_x);

    free(mat_strip);
    free(vec_strip);
    free(mat_displ);
    free(vec_displ);

    
    MPI_Finalize();

    return 0;
}

/*
    relative_error (with l2_norm) = |b - Ax|_2 / |b|_2
*/
double relative_error(int size, const double A[size][size], const double x[size], const double b[size]) {
    double *tmp;
    tmp = (double *) malloc (size * sizeof(double));
    matrix_vector_multiply(size, size, A, x, tmp);

    vector_sub(size, b, tmp, tmp);
    double res = pow (dot_product(size, tmp, tmp) / dot_product(size, b, b), 0.5); 
    free(tmp);
    return res;  
}

int read_matrix(int size, double matrix[size][size]) {
    FILE *fp_matrix;
    char matrix_file_name[] = "MATRIX";
    if ((fp_matrix = fopen(matrix_file_name, "r")) == NULL) {
        fprintf(stdout,"Can't open \"%s\" file.\n", matrix_file_name);
        return EXIT_FAILURE;     
    }
    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            fscanf(fp_matrix, "%lf ", &matrix[i][j]);
        }
        fscanf(fp_matrix, "\n");
    }
    if (fclose(fp_matrix) != 0) {
        fprintf(stderr,"Error closing file\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int read_vector(int size, double vector[size]) {
    FILE *fp_vector;
    char vector_file_name[] = "VECTOR";
    if ((fp_vector = fopen(vector_file_name, "r")) == NULL) {
        fprintf(stdout,"Can't open \"%s\" file.\n", vector_file_name);
        return EXIT_FAILURE;     
    }
    
    for(int i = 0; i < size; i++) {
        fscanf(fp_vector, "%lf ", &vector[i]);
    }
    if (fclose(fp_vector) != 0) {
        fprintf(stderr,"Error closing file\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void gen_symmetric_matrix(int size, double matrix[size][size]) {
    srand(time(NULL));

    for(int i = 0; i < size; i++) {
        for(int j = i; j < size; j++) {
            matrix[i][j] = 1;
            
            // make sure the generated matrix is positive definite
            if(i == j) {
                matrix[i][j] += size;    
            }
        }
    }

    for(int i = 0; i < size; i++) {
        for(int j = i - 1; j >= 0; j--) {
            matrix[i][j] = matrix[j][i];
        }
    }

}


void gen_vector(int size, double vector[size]) {
    srand(time(NULL));

    for(int i = 0; i < size; i++) { 
        vector[i] = 1;
    }

}

void seq_conj_grad(int size, const double A[size][size], const double b[size], double x[size]) {
    
    double *d, *g, *w, *vec_tmp;
    double g_dot_g, alpha, beta, scalar_tmp;

    d = (double *) malloc (size * sizeof(double));
    g = (double *) malloc (size * sizeof(double));
    w = (double *) malloc (size * sizeof(double));
    vec_tmp = (double *) malloc (size * sizeof(double));

    matrix_vector_multiply(size, size, A, x, vec_tmp);
    vector_sub(size, b, vec_tmp, d);
    vector_copy(size, d, g);

    free(vec_tmp);

    g_dot_g = dot_product(size, g, g);

    int k = 0;
    while (k < MAX_ITER && pow(g_dot_g, 0.5) > TOLERANCE) {
        matrix_vector_multiply(size, size, A, d, w);

        alpha = g_dot_g / dot_product(size, d, w);
        
        axpy(size, alpha, d, x);

        axpy(size, -1 * alpha, w, g);

        scalar_tmp = dot_product(size, g, g);
        beta = scalar_tmp / g_dot_g;
        
        axpy_save_z(size, beta, d, g, d);

        g_dot_g = scalar_tmp;

        k++;
    }

    free(d);
    free(g);
    free(w);
}

void print_matrix(int row, int col, const double matrix[row][col]) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

double dot_product(int size, const double vec_1[size], const double vec_2[size]) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += vec_1[i] * vec_2[i];
    }
    return res;
}

void vector_add(int size, const double vec_1[size], const double vec_2[size], double res[size]) {
    for(int i = 0; i < size; i++) {
        res[i] = vec_1[i] + vec_2[i];
    }
}


void vector_sub(int size, const double vec_1[size], const double vec_2[size], double res[size]) {
    for(int i = 0; i < size; i++) {
        res[i] = vec_1[i] - vec_2[i];
    }
}

void constant_multiply_vector(int size, const double vec[size], double multiplier, double res[size]) {
    for(int i = 0; i < size; i++) {
        res[i] = vec[i] * multiplier;
    }
}  

void matrix_vector_multiply(int row, int col, const double A[row][col], const double vec[col], double res[row]) {
    for(int i = 0; i < row; i++) {
        res[i] = 0;
        for(int j = 0; j < col; j++) {
            res[i] += A[i][j] * vec[j];
        }
    }
}


void vector_copy(int size, const double source[size], double dest[size]) {
    for(int i = 0; i < size; i++) {
        dest[i] = source[i];
    }
}

/*
    y = ax + y,
    x and y are vectors,
    a constant
*/
void axpy(int size, double a, const double x[size], double y[size]) {
    for (int i = 0; i < size; i++) {
        y[i] = y[i] + a*x[i];
    }
}


/*
    z = ax + y,
    x, z, and y are vectors,
    a constant
*/
void axpy_save_z(int size, double a, const double x[size], const double y[size], double z[size]) {
    for(int i = 0; i < size; i++) {
        z[i] = y[i] + a*x[i];
    }
}