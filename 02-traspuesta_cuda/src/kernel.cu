#include <cuda_runtime_api.h>
#include <cstdio>

#include "../include/kernel.cuh"

#define TAMAÑO_BLOQUE 16

#include "../include/toggles.h"
namespace CUDA {


    template<typename T>
    __global__ void transpose(const T *A, T * DST,Dimensiones dimA)
    {
        long fila = blockIdx.y * blockDim.y + threadIdx.y;
        long columna = blockIdx.x * blockDim.x + threadIdx.x;

        if(columna < dimA.width && fila < dimA.height)
        {
            DST[columna * dimA.height + fila] = A[columna + fila * dimA.width];
        }

    }
    void transponer_cuda(const float *A , float *DST, Dimensiones dimA, float * tiempoEjecucion)
    {
        //Matrices de la GPU
        float *A_GPU;
        float *DST_GPU;

        //Asignamos memoria a las matrices
        if(cudaMalloc(&A_GPU,sizeof (float) * dimA.width * dimA.height) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz A_GPU\n");
            return;
        }
        if(cudaMalloc(&DST_GPU,sizeof (float) * dimA.width * dimA.height) != cudaSuccess){
            printf("[ERROR] No se ha podido reservar memoria la matriz A_GPU\n");
            return;
        }

        //Y copiamos los valores de las matrices del host
        if(cudaMemcpy(A_GPU,A,sizeof (float) * dimA.width * dimA.height,cudaMemcpyHostToDevice) != cudaSuccess){
            printf("[ERROR] No se ha podido copiar el objeto A a A_GPU\n");
            return;
        }

        dim3 DimensionesBloque(TAMAÑO_BLOQUE,TAMAÑO_BLOQUE);
        dim3 DimensionesGrid( (dimA.width + DimensionesBloque.x - 1) / DimensionesBloque.x, (dimA.height + DimensionesBloque.x - 1) / DimensionesBloque.x);


#ifdef DEBUG
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
#endif
        transpose<<<DimensionesGrid,DimensionesBloque>>>(A_GPU,DST_GPU,dimA);

#ifdef DEBUG
        cudaEventRecord(stop);
#endif

        if(cudaMemcpy(DST,DST_GPU,sizeof(float) * dimA.width * dimA.height,cudaMemcpyDeviceToHost) != cudaSuccess){
            printf("[ERROR] No se ha podido copiar la matrix DST al host\n");
            return;
        }
#ifdef DEBUG
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(tiempoEjecucion,start,stop);
#endif

        //Esperamos que se terminen los procesos en la gpu
        cudaDeviceSynchronize();



        //Liberamos la memoria
        cudaFree(A_GPU);
        cudaFree(DST_GPU);
        
    }

}