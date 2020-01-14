#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ising_sequential.h"

#define N 600
#define K 100

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

	int *G;
	double w[25] = {0.004 , 0.016 , 0.026 , 0.016 , 0.004 ,
	                0.016 , 0.071 , 0.117 , 0.071 , 0.016 ,
					0.026 , 0.117 ,   0   , 0.117 , 0.026 ,
					0.016 , 0.071 , 0.117 , 0.071 , 0.016 ,
					0.004 , 0.016 , 0.026 , 0.016 , 0.004};

	// Allocate memory
	G = (int*)malloc(n*n*sizeof(int));
	if(G == NULL){
		printf("ERROR: Cannot allocate memory for G. Aborting...");
		return 1;
	}

    // Assign values to G
	FILE *fp;
    fp = fopen("conf-init.bin", "rb");
    if((k == 1 || k == 4 || k == 11) && (n == 517) && (fp != NULL)){
        // Assign values from file "conf-init.bin"
        fread(G, sizeof(int), n*n, fp);
    }
    else{
        // Assign random values to G
        int spin[] = {-1, 1};
        for(int i = 0; i < n; i++)
	      for(int j = 0; j < n; j++)
	        G[i*n + j] = spin[rand()%2];
    }

	// Execute function
    ising(G, w, k, n);

	// Validate results
	if((k == 1 || k == 4 || k == 11) && (n == 517) && (fp != NULL)){
        validate(G, k, n);
        fclose(fp);
    }
	else
	{
		printf("\nValidation: No validation provided.\n");
	}

	return 0;
}

void ising( int *G, double *w, int k, int n){

	// Capture start time
    clock_t start, end;
    double time_used;
	start = clock();

    int *H, *temp;
	double influence = 0.0;
	H = (int*)malloc(n*n*sizeof(int));
	for(int q = 0; q < k; q++){
       for(int i = 0; i < n; i++){
	   	   for(int j = 0; j < n; j++){
               for(int x = 0; x < 5; x++){
	   	   	       for(int y = 0; y < 5; y++){
	   	   	           influence += w[x*5 + y]*G[((i - 2 + x + n)%n)*n + ((j - 2 + y + n)%n)];
	   	   	       }
	            }
	   	        H[i*n + j] = G[i*n + j];
	   	        if(influence > 0.00000001){
	   	   	     H[i*n + j] = 1;
	   	        }
	   	        else if(influence < -0.00000001)
	   	        {
	   	   	     H[i*n + j] = -1;
	   	        }
	   	        influence = 0.0;
	   	   }
	   }
	   if( checkForNoChanges(H, G, n) ){
		   printf("\nNo changes: %d iterations\n", q);
		   break;
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

    // Capture end time
	end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime used: %f sec\n",time_used);

}

int checkForNoChanges(int *H, int *G, int n){
   for(int i = 0; i < n; i++){
     for(int j = 0; j < n; j++){
		 if(H[i*n + j] != G[i*n + j])
		   return 0;
	 }
   }
   return 1;
}


void validate(int *G, int k, int n){
	int *Gx, counter = 0;

	Gx = (int*)malloc(n*n*sizeof(int));

	FILE *fp;
	if(k == 1)
	  fp = fopen("conf-1.bin", "rb");
	else if(k == 4)
	  fp = fopen("conf-4.bin", "rb");
	else
	  fp = fopen("conf-11.bin", "rb");

    fread(Gx, sizeof(int), n*n, fp);
    fclose(fp);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(G[i*n + j] != Gx[i*n + j]){
				printf("\nWRONG");
				printf("\n%d %d",i, j);
				printf("\n%d   %d\n",G[i*n + j],Gx[i*n + j]);
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
