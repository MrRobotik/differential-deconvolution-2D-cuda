#ifndef OPTIMIZERKERNELS_H
#define OPTIMIZERKERNELS_H
#ifdef __CUDACC__

__global__ void zeroDifferential(
    float *d_imageDifferential,
    size_t imagePitch);

__global__ void evalObjectiveFnDerivative(
    const float *d_imageExpected,
    const float *d_imageObserved,
    const float *d_pointSpreadFnFlip,
    float *d_imageDifferential,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    size_t pointSpreadFnPitch);

__global__ void evalRegularizerDerivative(
    const float *d_imageIntrinsic,
    float *d_imageDifferential,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    float optimizerLambda);

__global__ void updateObserved(
    const float *d_imageDifferential,
    const float *d_pointSpreadFn,
    float *d_imageObserved,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    size_t pointSpreadFnPitch,
    float optimizerEta);

__global__ void updateIntrinsic(
    const float *d_imageDifferential,
    float *d_imageIntrinsic,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t imagePaddedPitch,
    float optimizerEta);

#endif
#endif
