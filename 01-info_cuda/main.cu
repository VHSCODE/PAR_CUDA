
#include "cuda_runtime_api.h"
#include "cuda.h"
#include <iostream>
int main(int argc, char * argv[])
{

    int devcount;
    cudaGetDeviceCount(&devcount);

    std::cout << "Cuda devices:" << std::endl;
    for(int i = 0; i < devcount;i++){

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props,i);

        std::cout << i + 1 << "/" << devcount << ": "<<  props.name << std::endl;
        std::cout << "----------------------------------" << std::endl;

        std::cout << "Memoria Total : " << props.totalGlobalMem / (1024 * 1024) << "Mb" << std::endl;
        std::cout << "Frecuencia de Reloj (GPU) : " <<  ( (float) props.clockRate / 1000000) << " GHz" <<  std::endl;
        std::cout << "Tamaño Warp : " << props.warpSize << std::endl;
        std::cout << "Threads por bloque : " << props.maxThreadsPerBlock << std::endl;
        std::cout << "Correccion de errores de memoria ? : " << (props.ECCEnabled ? "Si" : "No") << std::endl;
        std::cout << "Tamaño de cache L2 : " << props.l2CacheSize / 1024 << " KB" << std::endl;


    }
	return 0;
}
