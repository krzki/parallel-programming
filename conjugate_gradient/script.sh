#!/bin/bash

p=1


echo "Start Compiling"
mpicc conj_gradv2.c -o conj_gradv2.o -lm
mpicc mat_vec_generator.c -o mat_vec_generator.o
printf "Done Compilation\n\n"


rm result/$1.txt || true
echo "Start Matrix Generation"
mpirun --oversubscribe -np 4 mat_vec_generator.o $1
printf "Done Matrix Generation\n\n"

echo "Start running conjugate gradient"
while [ $p -le 4 ]
do
    mpirun --oversubscribe -np $p conj_gradv2.o $1 >> results/$1.txt
    p=$(( $p * 2 ))
done
printf "Done running conjugate gradient\n\n"