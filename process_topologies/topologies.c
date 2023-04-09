/*
    Program simulasi penggunaan process topologies pada MPI
*/

#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv)
{
    int comm_sz;

    int ndims = 2;
    int coords[ndims], dims[ndims], periods[ndims], source, dest, my_rank, reorder;
    MPI_Comm comm_2d;
    MPI_Status status;

    for(int i = 0; i < ndims; i++) {
        dims[i] = 0;
        periods[i] = 1;
    }
    reorder = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

    // distribute workers for each dimension (balanced)
    MPI_Dims_create(comm_sz, ndims, dims);

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_2d);

    // get cartesian coordinate for my_rank
    MPI_Cart_coords(comm_2d, my_rank, ndims, coords);
   
    MPI_Cart_shift(comm_2d, 0, coords[1], &source, &dest);

    float a,b;
    a = my_rank;

    MPI_Sendrecv (&a, 1, MPI_FLOAT, dest, 0, &b, 1, MPI_FLOAT,
        source, 0, comm_2d, &status);

    printf("Rank %2d, with coordinate (%d, %d), receives data %4.1f from Rank %2d\n",
            my_rank, coords[0], coords[1], b, source);
    
    MPI_Finalize();
}