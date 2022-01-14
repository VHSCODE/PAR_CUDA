#include <cstdio>
#include <cstring>
#include "include/kernel.cuh"
#include "include/MatrixUtils.h"
#include <chrono>
#include "include/toggles.h"


using clk = std::chrono::system_clock;
using sec = std::chrono::duration<double>;


void imprimir_matriz(const Matrix2D<float> *matrix);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: ./traspuesta_cuda a b \n");
        return 0;
    }


    float tiempo_ejecucion_cuda;

    Matrix2D<float> A{};
    Matrix2D<float> ResultadoSerie{};
    Matrix2D<float> ResultadoCuda{};


    //Dimensiones de las matrices
    long a, b;

    //Obtenemos las dimensiones
    a = strtol(argv[1], nullptr, 10);
    b = strtol(argv[2], nullptr, 10);



    MatrixUtils::gen_matriz(a,b,&A,100);
    
    ResultadoSerie.height = A.height;
    ResultadoSerie.width = A.width;
    ResultadoSerie.elements = (float *) malloc(sizeof(float) * A.height * A.width);

    ResultadoCuda.height = A.height;
    ResultadoCuda.width = A.width;
    ResultadoCuda.elements = (float *) malloc(sizeof(float) * A.height * A.width);

    #ifdef IMPRIMIR
    printf("[MATRIZ ORIGINAL]\n");
    imprimir_matriz(&A);
    #endif


    //#### SERIE ####
    //Comenzamos los calculos en la version serie

    printf("Calculando version serie...");

#ifdef DEBUG
    const auto before = clk::now();
#endif
    MatrixUtils::transponer(&A,&ResultadoSerie);

    printf("[OK]\n");

#ifdef DEBUG
    const sec tiempo_serie = clk::now() - before;
#endif

#ifdef IMPRIMIR
    printf("[MATRIZ SERIE]\n");
    imprimir_matriz(&ResultadoSerie);
#endif



    //#### CUDA ####

    printf("Calculando version CUDA...");
    //Comenzamos el proceso de cálculo en CUDA


    CUDA::transponer_cuda(A.elements,ResultadoCuda.elements,Dimensiones{A.height,A.width},&tiempo_ejecucion_cuda);

    ResultadoCuda.height = A.width;
    ResultadoCuda.width = A.height;

    printf("[OK]\n");

#ifdef IMPRIMIR
    printf("[MATRIZ CUDA]\n");
    imprimir_matriz(&ResultadoCuda);
#endif




#ifdef COMPROBAR

    //Comprobamos si el cálculo es correcto
    printf("Comprobando si el calculo es correcto...");

    //Usamos esta funcion debido a las comas, por lo que tendremos que especificar un valor de tolerencia, por defecto es de 0.01
    bool ret = MatrixUtils::son_iguales(&ResultadoSerie,&ResultadoCuda);

    if (!ret) {
        printf("[FAILED]: \nEl calculo CUDA NO es coherente con la version serie :(\n");
    } else
        printf("[OK]\nEl calculo CUDA SI es coherente con la version serie :)\n");

#endif


    //Liberamos la memoria utilizada
    free(A.elements);
    free(ResultadoSerie.elements);
    free(ResultadoCuda.elements);


#ifdef DEBUG

    printf("Tiempo Ejecucion Serie : %1.3f ms\n",tiempo_serie.count() * 1000);
    printf("Tiempo Ejecucion Cuda : %1.3f ms\n",tiempo_ejecucion_cuda);
#endif


    return 0;

}


void imprimir_matriz(const Matrix2D<float> *matrix) {

    int i, j;
    for (i = 0; i < matrix->height; i++) {
        for (j = 0; j < matrix->width; j++) {
            printf("%1.1f  ", matrix->elements[i * matrix->width + j]);
        }
        printf("\n");
    }
    printf("\n\n");


}