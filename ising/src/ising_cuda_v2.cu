#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ising_cuda.h"

#define N 1000
#define K 60
#define BLOCK_OF_MOMENTS 3

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void ising(int *G, int *H, double *w, int n){

    int i_start = (threadIdx.x + blockIdx.x * blockDim.x)*BLOCK_OF_MOMENTS;
    int j_start = (threadIdx.y + blockIdx.y * blockDim.y)*BLOCK_OF_MOMENTS;

    if( (i_start < n) && (j_start < n)){

        int i_end = i_start + BLOCK_OF_MOMENTS;
        int j_end = j_start + BLOCK_OF_MOMENTS;

        if(i_end > n)
          i_end = n;
        if(j_end > n)
          j_end = n;

        double influence = 0.0;
        for(int k = i_start; k < i_end; k++){
            for(int l = j_start; l < j_end; l++){
                for(int x = 0; x < 5; x++){
                    for(int y = 0; y < 5; y++){
                        influence += w[x*5 + y]*G[((k - 2 + x + n)%n)*n + ((l - 2 + y + n)%n)];
                    }
                }
                H[k*n + l] = G[k*n + l];
                if(influence > 0.000000001){
                    H[k*n + l] = 1;
                }
                else if(influence < -0.000000001)
                {
                    H[k*n + l] = -1;
                }
                influence = 0.0;
            }
        }
    }

}

int main(int argc, char** argv){
  // Declare all variables
	int n = 0;
int k = 0;
if (argc != 3)
{
		n = N;
		k = K;
}
else
{
		n = atoi(argv[1]);
		k = atoi(argv[2]);
		printf("Input n=%d k=%d", n, k);
}

    int *G, *G_final, *G_dev, *H_dev;
    double *w_dev;
	double w[25] = {0.004 , 0.016 , 0.026 , 0.016 , 0.004 ,
	                0.016 , 0.071 , 0.117 , 0.071 , 0.016 ,
					0.026 , 0.117 ,   0   , 0.117 , 0.026 ,
					0.016 , 0.071 , 0.117 , 0.071 , 0.016 ,
					0.004 , 0.016 , 0.026 , 0.016 , 0.004};

    // Allocate host memory
    G = (int*)malloc(n*n*sizeof(int));
    G_final = (int*)malloc(n*n*sizeof(int));
	if(G == NULL || G_final == NULL){
		printf("ERROR: Cannot allocate host memory. Aborting...");
		return 1;
	}

    // Allocate device memory
    HANDLE_ERROR( cudaMalloc((void**) &G_dev, n*n*sizeof(int))  );
    HANDLE_ERROR( cudaMalloc((void**) &H_dev, n*n*sizeof(int))  );
    HANDLE_ERROR( cudaMalloc((void**) &w_dev, 25*sizeof(double)));

    // Write to host memory (Assign random values to G)
    int spin[] = {-1, 1};
    for(int i = 0; i < n; i++)
	  for(int j = 0; j < n; j++)
	    G[i*n + j] = spin[rand()%2];

    // Copy host memory to device memory
    HANDLE_ERROR( cudaMemcpy(G_dev, G, n*n*sizeof(int), cudaMemcpyHostToDevice)  );
    HANDLE_ERROR( cudaMemcpy(w_dev, w, 25*sizeof(double), cudaMemcpyHostToDevice));

    printf("\nComputing...");

    // Capture start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Set kernel dimesions
    dim3 dimGrid((int) ceil((double)N/(double)(32*BLOCK_OF_MOMENTS)), (int) ceil((double)N/(double)(32*BLOCK_OF_MOMENTS)));
    dim3 dimBlock(32, 32);

    // Execute kernel on the device
    for(int q = 0; q < k; q++){
        if( q%2 == 0){
            ising<<< dimGrid, dimBlock >>>(G_dev, H_dev, w_dev, n);
        }
        else{
            ising<<< dimGrid, dimBlock >>>(H_dev, G_dev, w_dev, n);
        }
    }

    // Write GPU results to host memory
    if( k%2 == 1)
      HANDLE_ERROR( cudaMemcpy(G_final, H_dev, n*n*sizeof(int), cudaMemcpyDeviceToHost) );
    else
      HANDLE_ERROR( cudaMemcpy(G_final, G_dev, n*n*sizeof(int), cudaMemcpyDeviceToHost) );

    // Capture end time
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "\nTime used for parallel call:  %3.3f sec\n", elapsedTime*0.001 );

    // Free device memory
    HANDLE_ERROR( cudaFree(G_dev) );
    HANDLE_ERROR( cudaFree(H_dev) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    // Validate results
    printf("\nValidating...");
	validate(G, G_final, w, k, n);

    // Free host memory
    free(G);
    free(G_final);

	return 0;
}

void validate(int *G, int *G_final, double *w, int k, int n){
    int counter = 0;

    clock_t start, end;
    double time_used;
    start = clock();

    // Run sequential code for validation
    ising_host(G, w, k, n);

    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime used for sequential call: %f sec\n",time_used);

    for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(G[i*n + j] != G_final[i*n + j]){
                printf("\nWRONG");
				printf("\n%d %d",i, j);
				printf("\n%d   %d\n",G[i*n + j],G_final[i*n + j]);
				counter++;
			}
		}
	}
    if(counter == 0)
      printf("\nValidation: CORRECT\n");
	else
	{
	  printf("\nValidation: Wrong\n");
	  printf("\n%d wrong values\n",counter);
	}
}

void ising_host( int *G, double *w, int k, int n){
    int *H, *temp;
	double influence = 0.0;
    H = (int*)malloc(n*n*sizeof(int));
    if(H == NULL){
		printf("ERROR: Cannot allocate memory for H. Aborting...");
	}
	for(int q = 0; q < k; q++){
       for(int i = 0; i < n; i++){
	   	   for(int j = 0; j < n; j++){
               for(int x = 0; x < 5; x++){
	   	   	       for(int y = 0; y < 5; y++){
	   	   	           influence += w[x*5 + y]*G[((i - 2 + x + n)%n)*n + ((j - 2 + y + n)%n)];
	   	   	       }
	            }
	   	        H[i*n + j] = G[i*n + j];
	   	        if(influence > 0.000000001){
	   	   	     H[i*n + j] = 1;
	   	        }
	   	        else if(influence < -0.000000001)
	   	        {
	   	   	     H[i*n + j] = -1;
	   	        }
	   	        influence = 0.0;
	   	   }
	   }
	   temp = G;
	   G = H;
	   H = temp;
	}

	if(k%2 == 1){
		temp = G;
	    G = H;
	    H = temp;
		for(int i = 0; i < n; i++)
		  for(int j = 0; j < n; j++)
		     G[i*n + j] = H[i*n + j];
	}

}
