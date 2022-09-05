#include "OptimizerKernels.h"

__device__ static inline float *elementPtr(
    float *ptr,
    unsigned int row,
    unsigned int col,
    size_t pitch)
{
    char *cptr = reinterpret_cast<char *>(ptr);
    return reinterpret_cast<float *>(cptr + size_t(row) * pitch) + col;
}

__device__ static inline const float *elementPtr(
    const float *ptr,
    unsigned int row,
    unsigned int col,
    size_t pitch)
{
    const char *cptr = reinterpret_cast<const char *>(ptr);
    return reinterpret_cast<const float *>(cptr + size_t(row) * pitch) + col;
}

__device__ static inline const float *nextRowPtr(
    const float *ptr,
    size_t pitch)
{
    const char *cptr = reinterpret_cast<const char *>(ptr);
    return reinterpret_cast<const float *>(cptr + pitch);
}

__global__ void zeroDifferential(
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    float *dest = elementPtr(
        d_imageDifferential,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    *dest = 0.0f;
}

__global__ void evalObjectiveFnDerivative(
    const float *d_imageExpected,
    const float *d_imageObserved,
    const float *d_pointSpreadFnFlip,
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t pointSpreadFnPitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // initial offset
    d_imageExpected = elementPtr(d_imageExpected, row, col, imagePitch);
    d_imageObserved = elementPtr(d_imageObserved, row, col, imagePitch);

    // differentiate sum of squared differences
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float f_wrt_g = 2.0f * (d_imageObserved[j] - d_imageExpected[j]);
            float g_wrt_x = d_pointSpreadFnFlip[j];
            accumulator += f_wrt_g * g_wrt_x;
        }
        d_imageExpected =
            nextRowPtr(d_imageExpected, imagePitch);
        d_imageObserved =
            nextRowPtr(d_imageObserved, imagePitch);
        d_pointSpreadFnFlip =
            nextRowPtr(d_pointSpreadFnFlip, pointSpreadFnPitch);
    }
    // accumulate result
    float *dest = elementPtr(
        d_imageDifferential,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    *dest += accumulator;
}

__global__ void evalRegularizerDerivative(
    const float *d_imageIntrinsic,
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float regularizerLambda)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // read center value
    float valueCenter = *elementPtr(
        d_imageIntrinsic,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);

    // initial offset
    d_imageIntrinsic = elementPtr(
        d_imageIntrinsic,
        row + (imageRowPadding - 1),
        col + (imageColPadding - 1),
        imagePitch);

    // differentiate gamma corrected sum of squared differences (3x3 area)
    float accumulator = 0.0f;
    for (int i = 0; i < 3; i ++) {
        for (int j = 0; j < 3; j ++) {
            float value = d_imageIntrinsic[j];
            float denom = sqrtf(value * valueCenter);
            accumulator += 1.0f - (value / denom);
        }
        d_imageIntrinsic = nextRowPtr(d_imageIntrinsic, imagePitch);
    }
    // numerical problems
    if (__isnanf(accumulator) || __isinff(accumulator)) {
        return;
    }
    // accumulate result
    float *dest = elementPtr(
        d_imageDifferential,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    *dest += (1.0f / 8.0f) * regularizerLambda * accumulator;
}

__global__ void updateObserved(
    const float *d_imageDifferential,
    const float *d_pointSpreadFn,
    float *d_imageObserved,
    int imageRowPadding,
    int imageColPadding,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t pointSpreadFnPitch,
    float gradientDescentEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // initial offset
    d_imageDifferential = elementPtr(d_imageDifferential, row, col, imagePitch);

    // apply convolution change
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float deltaEnergy = -gradientDescentEta * d_imageDifferential[j];
            accumulator += deltaEnergy * d_pointSpreadFn[j];
        }
        d_imageDifferential = nextRowPtr(d_imageDifferential, imagePitch);
        d_pointSpreadFn = nextRowPtr(d_pointSpreadFn, pointSpreadFnPitch);
    }
    // accumulate result
    float *dest = elementPtr(
        d_imageObserved,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    *dest += accumulator;
}

__global__ void updateIntrinsic(
    const float *d_imageDifferential,
    float *d_imageIntrinsic,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float gradientDescentEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // energy change
    float diff = *elementPtr(
        d_imageDifferential,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    float deltaEnergy = -gradientDescentEta * diff;

    // accumulate result
    float *dest = elementPtr(
        d_imageIntrinsic,
        row + imageRowPadding,
        col + imageColPadding,
        imagePitch);
    *dest += deltaEnergy;
}
