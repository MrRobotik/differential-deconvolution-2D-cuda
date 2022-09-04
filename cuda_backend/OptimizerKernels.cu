#include "OptimizerKernels.h"

__device__ static void advancePtr(float **ptr, size_t offset)
{
    char *cptr = reinterpret_cast<char *>(*ptr);
    *ptr = reinterpret_cast<float *>(cptr + offset);
}

__device__ static void advancePtr(const float **ptr, size_t offset)
{
    const char *cptr = reinterpret_cast<const char *>(*ptr);
    *ptr = reinterpret_cast<const float *>(cptr + offset);
}

__global__ void zeroDifferential(
    float *d_imageDifferential,
    size_t imagePitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    advancePtr(&d_imageDifferential, row * imagePitch);
    d_imageDifferential[col] = 0.0f;
}

__global__ void evalObjectiveFnDerivate(
    const float *d_imageExpected,
    const float *d_imageObserved,
    const float *d_pointSpreadFnFlip,
    float *d_imageDifferential,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    size_t pointSpreadFnPitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // initial offset
    size_t offset = row * imagePaddedPitch + col * sizeof(float);
    advancePtr(&d_imageExpected, offset);
    advancePtr(&d_imageObserved, offset);

    // sum of squared differences derivative
    float objectiveFnDerivative = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float f_wrt_g = 2.0f * (d_imageObserved[j] - d_imageExpected[j]);
            float g_wrt_x = d_pointSpreadFnFlip[j];
            objectiveFnDerivative += f_wrt_g * g_wrt_x;
        }
        // advance pointers
        advancePtr(&d_imageExpected, imagePaddedPitch);
        advancePtr(&d_imageObserved, imagePaddedPitch);
        advancePtr(&d_pointSpreadFnFlip, pointSpreadFnPitch);
    }
    // accumulate result
    advancePtr(&d_imageDifferential, row * imagePitch);
    d_imageDifferential[col] += objectiveFnDerivative;
}

__global__ void updateObserved(
    const float *d_imageDifferential,
    const float *d_pointSpreadFn,
    float *d_imageObserved,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    size_t pointSpreadFnPitch,
    float optimizerEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int rowShifted = row + (pointSpreadFnRows >> 1);
    unsigned int colShifted = col + (pointSpreadFnCols >> 1);

    // respective energy change
    advancePtr(&d_imageDifferential, row * imagePitch);
    float delta = -optimizerEta * d_imageDifferential[col];

    // apply convolution change
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            accumulator += delta * d_pointSpreadFn[j];
        }
        // advance pointers
        advancePtr(&d_imageDifferential, imagePitch);
        advancePtr(&d_pointSpreadFn, pointSpreadFnPitch);
    }
    // accumulate result
    advancePtr(&d_imageObserved, rowShifted * imagePaddedPitch);
    d_imageObserved[colShifted] += accumulator;
}

__global__ void updateIntrinsic(
    const float *d_imageDifferential,
    float *d_imageIntrinsic,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    float optimizerEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int rowShifted = row + (pointSpreadFnRows >> 1);
    unsigned int colShifted = col + (pointSpreadFnCols >> 1);

    // respective energy change
    advancePtr(&d_imageDifferential, row * imagePitch);
    float delta = -optimizerEta * d_imageDifferential[col];

    // accumulate result
    advancePtr(&d_imageIntrinsic, rowShifted * imagePaddedPitch);
    d_imageIntrinsic[colShifted] += delta;
}
