
#include "MatrixUtils.h"

namespace CUDA {
    void matmuladd_calcular(const float *A, const float *B, const float *C, float *R, Dimensiones dimA, Dimensiones dimB, Dimensiones dimC);
    void ejemplo(const float *A, const float *B, const float *C, float *R,long m, long n, long k);

}