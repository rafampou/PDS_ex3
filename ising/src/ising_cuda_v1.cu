#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ising_cuda.h"

#define N 700
#define K 200


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void checkForNoChanges(int *G, int *H, int *checkForNoChanges, int n){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if( (i < n) && (j < n)){
        if( G[i*n + j] != H[i*n + j] ){
            checkForNoChanges[0] = 1;
        }
    }
}

__global__ void ising(int *G, int *H, double *w, int n){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if( (i < n) && (j < n)){
        double influence = 0.0;
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

    int *checkForNoChanges_SomeSpins;
    int *checkForNoChanges_AllSpins;
    int *checkForNoChanges_SomeSpins_dev;
    int *checkForNoChanges_AllSpins_dev;
    int iterations = k;

    // Allocate host memory
    G = (int*)malloc(n*n*sizeof(int));
    G_final = (int*)malloc(n*n*sizeof(int));
	if(G == NULL || G_final == NULL){
		printf("\nERROR: Cannot allocate host memory. Aborting...\n");
		return 1;
    }
    checkForNoChanges_SomeSpins = (int*)malloc(sizeof(int));
    checkForNoChanges_AllSpins = (int*)malloc(sizeof(int));

    // Allocate device memory
    HANDLE_ERROR( cudaMalloc((void**) &G_dev, n*n*sizeof(int))  );
    HANDLE_ERROR( cudaMalloc((void**) &H_dev, n*n*sizeof(int))  );
    HANDLE_ERROR( cudaMalloc((void**) &w_dev, 25*sizeof(double)));
    HANDLE_ERROR( cudaMalloc((void**) &checkForNoChanges_SomeSpins_dev, sizeof(int) ));
    HANDLE_ERROR( cudaMalloc((void**) &checkForNoChanges_AllSpins_dev,  sizeof(int) ));

    // Write to host memory
    /* Assign random values to G) */
    int spin[] = {-1, 1};
    for(int i = 0; i < n; i++)
	  for(int j = 0; j < n; j++)
        G[i*n + j] = spin[rand()%2];
    /* Assign values to checking variables */
    checkForNoChanges_SomeSpins[0] = 0;
    checkForNoChanges_AllSpins[0] = 0;

    // Copy host memory to device memory
    HANDLE_ERROR( cudaMemcpy(G_dev, G, n*n*sizeof(int), cudaMemcpyHostToDevice)  );
    HANDLE_ERROR( cudaMemcpy(w_dev, w, 25*sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(checkForNoChanges_SomeSpins_dev, checkForNoChanges_SomeSpins, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR( cudaMemcpy(checkForNoChanges_AllSpins_dev, checkForNoChanges_AllSpins, sizeof(int), cudaMemcpyHostToDevice));

    printf("\nComputing...\n");

    // Capture start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // Set kernels dimesions
    dim3 ising_Grid((int) ceil(((double)N)/32.0), (int) ceil(((double)N)/32.0));
    dim3 ising_Block(32, 32);
    dim3 checkForNoChanges_SomeSpins_Grid(2, 2);
    dim3 checkForNoChanges_Block(32, 32);
    dim3 checkForNoChanges_AllSpins_Grid((int) ceil(((double)N)/32.0), (int) ceil(((double)N)/32.0));

    // Execute kernel on the device
    for(int q = 0; q < k; q++){
        if( q%2 == 0){
            ising<<< ising_Grid, ising_Block >>>(G_dev, H_dev, w_dev, n);

            // Check if no changes are made
            checkForNoChanges<<< checkForNoChanges_SomeSpins_Grid, checkForNoChanges_Block>>>(G_dev, H_dev, checkForNoChanges_SomeSpins_dev, n);
            HANDLE_ERROR( cudaMemcpy(checkForNoChanges_SomeSpins, checkForNoChanges_SomeSpins_dev, sizeof(int), cudaMemcpyDeviceToHost) );
            if(checkForNoChanges_SomeSpins[0] == 0){
               checkForNoChanges<<< checkForNoChanges_AllSpins_Grid, checkForNoChanges_Block>>>(G_dev, H_dev, checkForNoChanges_AllSpins_dev, n);
               HANDLE_ERROR( cudaMemcpy(checkForNoChanges_AllSpins, checkForNoChanges_AllSpins_dev, sizeof(int), cudaMemcpyDeviceToHost) );
               if( checkForNoChanges_AllSpins[0] == 0){
                   printf("\nNo changes: %d iterations\n", q);
                   iterations = q;
                   break;
               }
               else{
                   checkForNoChanges_AllSpins[0] = 0;
                   HANDLE_ERROR( cudaMemcpy(checkForNoChanges_AllSpins_dev, checkForNoChanges_AllSpins, sizeof(int), cudaMemcpyHostToDevice));
               }
            }
            else{
                checkForNoChanges_SomeSpins[0] = 0;
                HANDLE_ERROR( cudaMemcpy(checkForNoChanges_SomeSpins_dev, checkForNoChanges_SomeSpins, sizeof(int), cudaMemcpyHostToDevice));
            }
        }
        else{
            ising<<< ising_Grid, ising_Block >>>(H_dev, G_dev, w_dev, n);

            // Check if no changes are made
            checkForNoChanges<<< checkForNoChanges_SomeSpins_Grid, checkForNoChanges_Block>>>(G_dev, H_dev, checkForNoChanges_SomeSpins_dev, n);
            HANDLE_ERROR( cudaMemcpy(checkForNoChanges_SomeSpins, checkForNoChanges_SomeSpins_dev, sizeof(int), cudaMemcpyDeviceToHost) );
            if(checkForNoChanges_SomeSpins[0] == 0){
               checkForNoChanges<<< checkForNoChanges_AllSpins_Grid, checkForNoChanges_Block>>>(G_dev, H_dev, checkForNoChanges_AllSpins_dev, n);
               HANDLE_ERROR( cudaMemcpy(checkForNoChanges_AllSpins, checkForNoChanges_AllSpins_dev, sizeof(int), cudaMemcpyDeviceToHost) );
               if( checkForNoChanges_AllSpins[0] == 0){
                   printf("\nNo changes: %d iterations\n", q);
                   iterations = q;
                   break;
               }
               else{
                   checkForNoChanges_AllSpins[0] = 0;
                   HANDLE_ERROR( cudaMemcpy(checkForNoChanges_AllSpins_dev, checkForNoChanges_AllSpins, sizeof(int), cudaMemcpyHostToDevice));
               }
            }
            else{
                checkForNoChanges_SomeSpins[0] = 0;
                HANDLE_ERROR( cudaMemcpy(checkForNoChanges_SomeSpins_dev, checkForNoChanges_SomeSpins, sizeof(int), cudaMemcpyHostToDevice));
            }
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
    HANDLE_ERROR( cudaFree(checkForNoChanges_SomeSpins_dev));
    HANDLE_ERROR( cudaFree(checkForNoChanges_AllSpins_dev));
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    // Validate results
    validate(G, G_final, w, iterations, n);


    // Free host memory
    free(G);
    free(G_final);
    free(checkForNoChanges_SomeSpins);
    free(checkForNoChanges_AllSpins);

	return 0;
}

void validate(int *G, int *G_final, double *w, int k, int n){

    printf("\nValidating...\n");

    int counter = 0;

    clock_t start, end;
    double time_used;
    start = clock();

    // Run sequential code
    ising_sequential(G, w, k, n);

    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime used for sequential call: %3.3f sec\n",time_used);

    // Validate
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

void ising_sequential( int *G, double *w, int k, int n){
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
