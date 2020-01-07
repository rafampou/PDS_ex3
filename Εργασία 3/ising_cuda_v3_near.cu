#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ising_cuda.h"

#define N 50
#define K 50
#define BLOCK_OF_MOMENTS 3

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void ising(int *G, int *H, double *w, int n){

    __shared__ int sharedMoments[(32*BLOCK_OF_MOMENTS + 4)*(32*BLOCK_OF_MOMENTS + 4)];
   

    for(int k = threadIdx.x; k < 32*BLOCK_OF_MOMENTS + 4 ; k = k + 32){
        for(int l = threadIdx.y; l < 32*BLOCK_OF_MOMENTS + 4 ; l = l + 32){
            if( ((k + 96*blockIdx.x) < (n + 4)) && ((l + 96*blockIdx.y) < (n + 4)))
              sharedMoments[k*(32*BLOCK_OF_MOMENTS + 4) + l] = G[((k + 96*blockIdx.x - 2  + n)%n)*n + ((l + 96*blockIdx.y - 2 + n)%n)];
        }
    }  

    
    __syncthreads();

    
    int i_start = threadIdx.x*BLOCK_OF_MOMENTS; 
    int j_start = threadIdx.y*BLOCK_OF_MOMENTS;
    
    if(blockIdx.x*96 + i_start < n && blockIdx.y*96 + j_start < n){
    
        int i_end = i_start + BLOCK_OF_MOMENTS;
        int j_end = j_start + BLOCK_OF_MOMENTS;
    
        if(blockIdx.x*96 + i_end > n)
          if(blockIdx.x != 0)
            i_end = n%(blockIdx.x*96);
          else
            i_end = n;
        if(blockIdx.y*96 + j_end > n)
          if(blockIdx.y != 0)
            j_end = n%(blockIdx.y*96);
          else
            j_end = n;
        
        double influence = 0.0;
        for(int k = i_start; k < i_end; k++){
            for(int l = j_start; l < j_end; l++){
                for(int x = 0; x < 5; x++){
                    for(int y = 0; y < 5; y++){   
                        influence += w[x*5 + y]*sharedMoments[((k + 2 - 2 + x + 100)%100)*100 + ((l + 2 - 2 + y + 100)%100)];  
                    }
                } 
                H[(blockIdx.x*96 + k)*n + (blockIdx.y*96 + l)] = sharedMoments[(k + 2)*100 + (l + 2)]; 
                if(influence > 0.000000001){
                    H[(blockIdx.x*96 + k)*n + (blockIdx.y*96 + l)] = 1;
                }
                else if(influence < -0.000000001)
                {
                    H[(blockIdx.x*96 + k)*n + (blockIdx.y*96 + l)] = -1;
                }
                influence = 0.0;
            }
        }        
    }
}

int main(){    
    // Declare all variables
    int n = N, k = K;
	
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

    printf("\nComputing...\n");

    // capture start time
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