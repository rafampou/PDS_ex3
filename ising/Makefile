# ####################################################################
#
#			   C/C++ Makefile
#
# define the shell to bash
SHELL := /bin/bash

# define compile-time flags
CFLAGS = -O2 -lm

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above
#


lib: ising_cuda_v1.o ising_cuda_v2.o ising_cuda_v3.o ising_sequential.o
	ar rcs lib/ising_sequential.a lib/ising_sequential.o
	ar rcs lib/ising_cuda_v1.a lib/ising_cuda_v1.o
	ar rcs lib/ising_cuda_v2.a lib/ising_cuda_v2.o
	ar rcs lib/ising_cuda_v3.a lib/ising_cuda_v3.o

ising_sequential.o:
	gcc -Iinc $(CFLAGS) -c  src/ising_sequential.c -o lib/ising_sequential.o  -lm
	gcc -Iinc $(CFLAGS)  src/ising_sequential.c -o ising_sequential  -lm
ising_cuda_v1.o:
		nvcc -Iinc $(CFLAGS) -c  src/ising_cuda_v1.cu -o lib/ising_cuda_v1.o  -lm
		nvcc -Iinc $(CFLAGS)  src/ising_cuda_v1.cu -o ising_cuda_v1  -lm
ising_cuda_v2.o:
		nvcc -Iinc $(CFLAGS) -c  src/ising_cuda_v2.cu -o lib/ising_cuda_v2.o  -lm
		nvcc -Iinc $(CFLAGS)   src/ising_cuda_v2.cu -o ising_cuda_v2  -lm
ising_cuda_v3.o:
		nvcc -Iinc $(CFLAGS) -c  src/ising_cuda_v3.cu -o lib/ising_cuda_v3.o  -lm
		nvcc -Iinc $(CFLAGS)   src/ising_cuda_v3.cu -o ising_cuda_v3  -lm
clean:
	rm lib/*
