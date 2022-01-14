//
// Created by v on 4/1/22.
//

//Aqui se encuentran varias funciones para el manejo de matrices 2D en c++


#ifndef MATMUL_CUDA_MATRIXGEN_H
#define MATMUL_CUDA_MATRIXGEN_H

#include <cmath>
#include <vector>

struct Dimensiones{
    long height;
    long width;
};

template<typename T>
struct Matrix2D{
    long height;
    long width;
    T * elements;
};


class MatrixUtils {

public:
    template<typename T>
    static bool gen_matrices(long a, long b, long c, Matrix2D<T> *A, Matrix2D<T> *B, Matrix2D<T>*C, int max_val);


    template<typename T>
    static bool gen_matriz(long a, long b, Matrix2D<T> *A,int max_val);
    
    template<typename T>
    static bool gen_matrices_val(long a, long b, long c, Matrix2D<T> *A, Matrix2D<T> *B, Matrix2D<T>*C, T val);


    template<typename T>
    static bool son_iguales(const Matrix2D<T> *A, const Matrix2D<T> *B);


    inline static bool son_casi_iguales(const Matrix2D<float> *A, const Matrix2D<float> *B, float epsilon);


    template<typename T>
    static void multiplicar(const Matrix2D<T> *A, const Matrix2D<T> *B, Matrix2D<T> *DST);

    template<typename T>
    static void sumar(const Matrix2D<T> *A, const Matrix2D<T> *B, Matrix2D<T> *DST);

    template<typename T>
    static void transponer(const Matrix2D<T> *A, Matrix2D<T> *DST);


};


template<typename T>
bool MatrixUtils::son_iguales(const Matrix2D<T> *A, const Matrix2D<T> *B)
{
    if (A->height != B->height || A->width != B->width) {
        printf("Las dimensiones de las matrices a comparar  son incompatibles\n");
        return false;
    }
    for (long i = 0; i < A->height; i++) {
        for (long j = 0; j < A->width; j++) {
            if (A->elements[i * A->width + j] != B->elements[i * B->width + j])
                return false;
        }
    }

    return true;
}


template<typename T>
bool MatrixUtils::gen_matrices(long a, long b, long c, Matrix2D<T> *A, Matrix2D<T> *B, Matrix2D<T>*C, int max_val) {
    if (a <= 0 || b <= 0 || c <= 0) {
        printf("Introduce un valor valido para las dimensiones de las matrices\n");
        return false;
    }


    //Asignamos memoria a las matrices y comprobamos que la asignacion de memoria sea exitosa

    A->elements = (T *) malloc(sizeof(T *) * a * b);
    if (!A->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz A\n");
        return false;
    }
    A->height = a;
    A->width = b;


    B->elements = (T *) malloc(sizeof(T *) * b * c);
    if (!B->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz B\n");
        return false;
    }
    B->height = b;
    B->width = c;


    C->elements = (T *) malloc(sizeof(T *) * a * c);
    if (!C->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz C\n");
        return false;
    }
    C->height = a;
    C->width = c;
    int i, j;

    //Rellenamos las matrices con valores aleatorios
    for (i = 0; i < a; i++)
        for (j = 0; j < b; j++)
            A->elements[i * b + j] = (T) rand() / (T) (RAND_MAX / max_val + 1);

    for (i = 0; i < b; i++)
        for (j = 0; j < c; j++)
            B->elements[i * c + j] = (T) rand() / (T) (RAND_MAX / max_val + 1);

    for (i = 0; i < a; i++)
        for (j = 0; j < c; j++)
            C->elements[i * c + j] = (T) rand() / (T) (RAND_MAX / max_val + 1);

    return true;
}

template<typename T>
void MatrixUtils::multiplicar(const Matrix2D<T> *A, const Matrix2D<T> *B, Matrix2D<T> *DST){
    if (A->width != B->height) {
        printf("[ERROR] Las dimensiones de las matrices a multiplicar son incompatibles\n");
        return;
    }


    T sum = 0;
    for (long i = 0; i < A->height; i++) {
        for (long j = 0; j < B->width; j++) {
            for (long k = 0; k < A->width; k++) {

                sum += A->elements[i * A->width + k] * B->elements[k * B->width + j];
            }

            DST->elements[i * B->width + j] = sum;
            sum = 0;
        }
    }

}

template<typename T>
bool MatrixUtils::gen_matrices_val(long a, long b, long c, Matrix2D<T> *A, Matrix2D<T> *B, Matrix2D<T>*C, const T val) {
    if (a <= 0 || b <= 0 || c <= 0) {
        printf("Introduce un valor valido para las dimensiones de las matrices\n");
        return false;
    }


    //Asignamos memoria a las matrices y comprobamos que la asignacion de memoria sea exitosa

    A->elements = (T *) malloc(sizeof(T*) * a * b);
    if (!A->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz A\n");
        return false;
    }
    A->height = a;
    A->width = b;


    B->elements = (T *) malloc(sizeof(T*) * b * c);
    if (!B->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz B\n");
        return false;
    }
    B->height = b;
    B->width = c;


    C->elements = (T *) malloc(sizeof(T*) * a * c);
    if (!C->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz C\n");
        return false;
    }
    C->height = a;
    C->width = c;


    int i, j;

    //Rellenamos las matrices con valores 
    for (i = 0; i < a; i++)
        for (j = 0; j < b; j++)
            A->elements[i * b + j] = val;

    for (i = 0; i < b; i++)
        for (j = 0; j < c; j++)
            B->elements[i * c + j] = val;

    for (i = 0; i < a; i++)
        for (j = 0; j < c; j++)
            C->elements[i * c + j] = val;

    return true;

}

template<typename T>
void MatrixUtils::sumar(const Matrix2D<T> *A, const Matrix2D<T> *B, Matrix2D<T> *DST) {

    if (A->height != B->height || A->width != B->width) {
        printf("Las dimensiones de las matrices a sumar son incompatibles\n");
        return;
    }

    for (int i = 0; i < A->height; i++)
        for (int j = 0; j < A->width; j++)
            DST->elements[i * A->width + j] = A->elements[i * A->width + j] + B->elements[i * B->width + j];
}

bool MatrixUtils::son_casi_iguales(const Matrix2D<float> *A, const Matrix2D<float> *B, const float epsilon = 0.f)
{

    if (A->height != B->height || A->width != B->width) {
        printf("Las dimensiones de las matrices a comparar  son incompatibles\n");
        return false;
    }
    struct Par{
        float a;
        float b;
    };

    std::vector<Par> valores_anomalos;


    bool casi_iguales = true;

    for (long i = 0; i < A->height; i++) {
        for (long j = 0; j < A->width; j++) {
            if (std::fabs(A->elements[i * A->width + j] - B->elements[i * B->width + j]) >= epsilon)
            {
                valores_anomalos.push_back(Par{A->elements[i * A->width + j],B->elements[i * B->width + j]});
                casi_iguales = false;
            }

        }
    }


    if(!valores_anomalos.empty()) {
        printf("Valores anomalos en la comparacion:\n");
        for (auto par: valores_anomalos) {
            printf("x = %1.1f -- y = %1.1f -- Diff = %1.1f\n",par.a,par.b, std::abs(par.a - par.b));
        }
    }
    return casi_iguales;

}




template<typename T>
 void MatrixUtils::transponer(const Matrix2D<T> *A, Matrix2D<T> *DST){

    for(int i = 0; i < A->height; i++){
        for(int j = 0; j < A->width; j++){
            DST->elements[ j * A->height + i ] = A->elements[i * A->width + j];
        }
    }
    DST->height = A->width;
    DST->width = A->height;
}

template<typename T>
bool MatrixUtils::gen_matriz(long a, long b, Matrix2D<T> *A,int max_val)
{
    if (a <= 0 || b <= 0) {
        printf("Introduce un valor valido para las dimensiones de las matriz\n");
        return false;
    }


    //Asignamos memoria a las matrices y comprobamos que la asignacion de memoria sea exitosa

    A->elements = (T *) malloc(sizeof(T*) * a * b);
    if (!A->elements) {
        printf("[ERROR] Error al asignar memoria a la matriz A\n");
        return false;
    }
    A->height = a;
    A->width = b;

    for (int i = 0; i < a; i++)
        for (int j = 0; j < b; j++)
            A->elements[i * b + j] = (T) rand() / (T) (RAND_MAX / max_val + 1);

}


#endif //MATMUL_CUDA_MATRIXGEN_H