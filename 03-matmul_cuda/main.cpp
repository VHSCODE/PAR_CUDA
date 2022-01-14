#include <cstdio>
#include "include/kernel.cuh"
#include "include/MatrixUtils.h"

#define DEBUG
#define COMPROBAR
//#define IMPRIMIR


void imprimir_matriz(const Matrix2D<float> *matrix);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Uso: ./matmul_cuda a b c\n");
        return 0;
    }
    Matrix2D<float> A{};
    Matrix2D<float> B{};
    Matrix2D<float> C{};
    Matrix2D<float> R{};



    //Dimensiones de las matrices
    long a, b, c;

    //Obtenemos las dimensiones
    a = strtol(argv[1], nullptr, 10);
    b = strtol(argv[2], nullptr, 10);
    c = strtol(argv[3], nullptr, 10);


    int exito = MatrixUtils::gen_matrices(a, b, c, &A, &B, &C,100);


    //int exito = MatrixUtils::gen_matrices_val(a, b, c, &A, &B, &C, 1.0f);
    if (!exito) {
        return 0;
    }

    //Calculamos primero la version serie

    Matrix2D<float> R_SERIE{};
    R_SERIE.elements = (float *) malloc(sizeof(float) * a * c);

    R_SERIE.height = a;
    R_SERIE.width = c;

    if (!R_SERIE.elements) {
        printf("[ERROR] Error al asignar memoria a la matriz R_SERIE\n");

        return 0;
    }

    //#### SERIE ####

    //Comenzamos los calculos en la version serie

    printf("Calculando version serie...");
    MatrixUtils::multiplicar(&A, &B, &R_SERIE);
    //MatrixUtils::sumar(&R_SERIE,&C,&R_SERIE);
    printf("[OK]\n");

#ifdef IMPRIMIR
    printf("[MATRIZ SERIE]\n");
    imprimir_matriz(&R_SERIE);
#endif

    //#### CUDA ####
    R.elements = (float *) malloc(sizeof(float) * a * c);
    R.height = a;
    R.width = c;
    if (!R.elements) {
        printf("[ERROR] Error al asignar memoria a la matriz R\n");
        return 0;
    }

    printf("Calculando version CUDA...");
    //Comenzamos el proceso de cálculo en CUDA
    CUDA::matmuladd_calcular(A.elements, B.elements, C.elements, R.elements,
                             Dimensiones{A.height,A.width},
                             Dimensiones{B.height, B.width},
                             Dimensiones{C.height,C.width});


    printf("[OK]\n");

#ifdef IMPRIMIR
    printf("[MATRIZ CUDA]\n");
    imprimir_matriz(&R);
#endif




#ifdef COMPROBAR

    //Comprobamos si el cálculo es correcto
    printf("Comprobando si el calculo es correcto...");

    //Usamos esta funcion debido a las comas, por lo que tendremos que especificar un valor de tolerencia, por defecto es de 0.01
    bool ret = MatrixUtils::son_casi_iguales(&R_SERIE, &R);

    if (!ret) {
        printf("[FAILED]: \nEl calculo CUDA NO es coherente con la version serie\n");
        printf("Esto no significa que los datos sean erroneos, las operaciones de coma flotante pueden variar entre ejecuciones\n");
    } else
        printf("[OK]\nEl calculo CUDA SI es coherente con la version serie :)\n");

#endif


    //Liberamos la memoria utilizada
    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(R.elements);
    free(R_SERIE.elements);


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