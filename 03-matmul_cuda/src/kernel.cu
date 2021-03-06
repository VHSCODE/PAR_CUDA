#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>

#include "../include/kernel.cuh"

#include "../include/toggles.h"
#define TAMAÑO_BLOQUE 16
#include "../include/MatrixUtils.h"
namespace CUDA {


    template<typename T>
    __global__ void multiplicar_cuda(const T *A, const T *B, T * DST,Dimensiones dimA, Dimensiones dimB, Dimensiones dimDST)
    {
        long fila = blockIdx.y * blockDim.y + threadIdx.y;
        long columna = blockIdx.x * blockDim.x + threadIdx.x;

        T suma = 0;

        if(columna < dimB.width && fila < dimA.height){

            for(long i = 0; i < dimA.width; i++){

                suma += A[fila * dimA.width + i] * B[i * dimB.width + columna];
            }
            DST[fila * dimDST.width + columna] =  suma;
        }

    }

    template<typename T>
    __global__ void sumar_cuda(const T *A, const T *B, T *DST,long N)
    {
        long tid = (blockDim.x * blockIdx.x) + threadIdx.x;
        if (tid < N) {
            DST[tid] = A[tid] + B[tid];
        }
    }


    void matmuladd_calcular(const float *A, const float *B, const float *C, float *R, Dimensiones dimA, Dimensiones dimB, Dimensiones dimC,float *tiempoEjecucion)
    {

        if (dimA.width != dimB.height) {
            printf("[ERROR] Las dimensiones introducidas no son validas para hacer el calculo\n");

            return;
        }

        //Matrices de la GPU
        float *A_GPU, *B_GPU, *C_GPU, *MUL_GPU, * R_GPU;


        //Asignamos memoria a las matrices
        if(cudaMalloc(&A_GPU,sizeof (float) * dimA.width * dimA.height) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz A_GPU\n");
            return;
        }
        if(cudaMalloc(&B_GPU,sizeof (float) * dimB.width * dimB.height) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz B_GPU\n");
            return;
        }
        if(cudaMalloc(&C_GPU,sizeof (float) * dimC.width * dimC.height) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz C_GPU\n");
            return;
        }

        Dimensiones dimR{dimA.height, dimB.width};

        if(cudaMalloc(&MUL_GPU, sizeof (float) * dimR.height * dimR.width) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz MUL_GPU\n");
            return;
        }
        if(cudaMalloc(&R_GPU,sizeof (float) * dimR.height * dimR.width) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz C_GPU\n");
            return;
        }



        //Y copiamos los valores de las matrices del host
        if(cudaMemcpy(A_GPU,A,sizeof (float) * dimA.width * dimA.height,cudaMemcpyHostToDevice) != cudaSuccess){
            printf("[ERROR] No se ha podido copiar el objeto A a A_GPU\n");
            return;
        }


        if(cudaMemcpy(B_GPU,B,sizeof (float) * dimB.width * dimB.height,cudaMemcpyHostToDevice) != cudaSuccess){
            printf("[ERROR] No se ha podido copiar el objeto A a B_GPU\n");
            return;
        }
        if(cudaMemcpy(C_GPU,C,sizeof (float) * dimC.width * dimC.height,cudaMemcpyHostToDevice) != cudaSuccess){
            printf("[ERROR] No se ha podido copiar el objeto A a C_GPU\n");
            return;
        }

        //Creamos las dimensiones del grid y de los bloques
        dim3 dimensionesBloque(TAMAÑO_BLOQUE, TAMAÑO_BLOQUE);

        dim3 dimensionesGrid((dimB.height + TAMAÑO_BLOQUE - 1)/TAMAÑO_BLOQUE, (dimA.width+ TAMAÑO_BLOQUE-1)/TAMAÑO_BLOQUE);

        long N = dimC.width * dimC.height;

#ifdef DEBUG
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
#endif
        // Y comenzamos a llamar a los kernels
        multiplicar_cuda<<<dimensionesGrid,dimensionesBloque>>>(A_GPU, B_GPU, MUL_GPU,
                                                                dimA, dimB, dimR);


        dim3 dimGrid((N + TAMAÑO_BLOQUE -1 ) / TAMAÑO_BLOQUE);
        sumar_cuda<<<dimGrid, dimensionesBloque>>>(C_GPU,MUL_GPU, R_GPU,N );


#ifdef DEBUG
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
#endif
        //Esperamos que se terminen los procesos en la gpu
        cudaDeviceSynchronize();
#ifdef DEBUG


        cudaEventElapsedTime(tiempoEjecucion,start,stop);
#endif


        auto err = cudaMemcpy(R, R_GPU, sizeof(float) * dimR.height * dimR.width, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        //Escribimos el resultado en la matriz host
        if ( err != cudaSuccess)
        {

            printf("[ERROR] Error al copiar la matriz R de la GPU al host : %s \n", cudaGetErrorName(err));
            return;
        }




        //Liberamos la memoria
        cudaFree(A_GPU);
        cudaFree(B_GPU);
        cudaFree(C_GPU);
        cudaFree(MUL_GPU);
        cudaFree(R_GPU);
    }

}