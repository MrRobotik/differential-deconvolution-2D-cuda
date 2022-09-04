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

__global__ void evalObjectiveFnDerivative(
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

    // differentiate sum of squared differences
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float f_wrt_g = 2.0f * (d_imageObserved[j] - d_imageExpected[j]);
            float g_wrt_x = d_pointSpreadFnFlip[j];
            accumulator += f_wrt_g * g_wrt_x;
        }
        // advance
        advancePtr(&d_imageExpected, imagePaddedPitch);
        advancePtr(&d_imageObserved, imagePaddedPitch);
        advancePtr(&d_pointSpreadFnFlip, pointSpreadFnPitch);
    }
    // accumulate result
    advancePtr(&d_imageDifferential, row * imagePitch);
    d_imageDifferential[col] += accumulator;
}

__global__ void evalRegularizerDerivative(
    const float *d_imageIntrinsic,
    float *d_imageDifferential,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    float optimizerLambda)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int rowCenter = row + (pointSpreadFnRows >> 1);
    unsigned int colCenter = col + (pointSpreadFnCols >> 1);
    unsigned int rowPatch = rowCenter - 1u;
    unsigned int colPatch = colCenter - 1u;

    // read center value
    const float *temp = d_imageIntrinsic;
    advancePtr(&temp, rowCenter * imagePaddedPitch);
    float valueCenter = temp[colCenter];

    // initial offset
    size_t offset = rowPatch * imagePaddedPitch + colPatch * sizeof(float);
    advancePtr(&d_imageIntrinsic, offset);

    // differentiate gamma corrected sum of squared differences (3x3 area)
    float accumulator = 0.0f;
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            float value = d_imageIntrinsic[j];
            float denom = sqrtf(max(0.0f, value * valueCenter));
            accumulator += 1.0f - (value / denom);
        }
        // advance
        advancePtr(&d_imageIntrinsic, imagePaddedPitch);
    }
    // accumulate result
    advancePtr(&d_imageDifferential, row * imagePitch);
    d_imageDifferential[col] += optimizerLambda * accumulator;
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
    unsigned int rowCenter = row + (pointSpreadFnRows >> 1);
    unsigned int colCenter = col + (pointSpreadFnCols >> 1);

    // // energy change
    // advancePtr(&d_imageDifferential, row * imagePitch);
    // float delta = -optimizerEta * d_imageDifferential[col];

    // initial offset
    size_t offset = row * imagePitch + col * sizeof(float);
    advancePtr(&d_imageDifferential, offset);

    // apply convolution change
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float delta = -optimizerEta * d_imageDifferential[j];
            accumulator += delta * d_pointSpreadFn[j];
        }
        // advance
        advancePtr(&d_imageDifferential, imagePitch);
        advancePtr(&d_pointSpreadFn, pointSpreadFnPitch);
    }
    // accumulate result
    advancePtr(&d_imageObserved, rowCenter * imagePaddedPitch);
    d_imageObserved[colCenter] += accumulator;
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
    unsigned int rowCenter = row + (pointSpreadFnRows >> 1);
    unsigned int colCenter = col + (pointSpreadFnCols >> 1);

    // energy change
    advancePtr(&d_imageDifferential, row * imagePitch);
    float delta = -optimizerEta * d_imageDifferential[col];

    // accumulate result
    advancePtr(&d_imageIntrinsic, rowCenter * imagePaddedPitch);
    d_imageIntrinsic[colCenter] += delta;
}
