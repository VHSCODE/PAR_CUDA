#include <stdio.h>

#include <unistd.h>
#include <stdlib.h>

int main(int argc, char* argv[]){



	int n,m;
	n = 10;
	m = 10;
	float * C = malloc(n * m * sizeof(float));
	
	float *resultado = malloc(n * m * sizeof(float));
	

	//Rellenamos la matriz C con valores aleatorios.
	
	int i,j;
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j ++)
			C[i * n + j] = rand() % 101;

	

	//Realizamos la transposicion
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j ++)
			resultado[j * n + i] = C[i * n + j];



	printf("Matriz Original:\n");

	for(i = 0; i < n; i++){
		for(j = 0; j < m; j ++){
			printf("%1.1f  ", C[i * n + j]);
		}
		printf("\n");
	}
			

	printf("Matriz Traspuesta:\n");

	for(i = 0; i < n; i++){
		for(j = 0; j < m; j ++){
			printf("%1.1f  ", resultado[i * n + j]);
		}
		printf("\n");
	}


	free(C);
	free(resultado);
	return 0;

}
