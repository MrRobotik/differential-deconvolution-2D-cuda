#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifndef __CUDACC__
#include <stddef.h>
#include <include/cuda_runtime.h>
#endif

class Optimizer
{
public:

    Optimizer() = delete;

    Optimizer(
        const float *h_imageExpected,
        const float *h_imageObserved,
        const float *h_pointSpreadFn,
        const float *h_pointSpreadFnFlip,
        int imageRows,
        int imageCols,
        int pointSpreadFnRows,
        int pointSpreadFnCols);

    ~Optimizer();

    void step(double optimizerEta, double optimizerLambda);

    void getResultFromDevice(float *h_imageIntrinsic) const;

private:

    // device metadata
    int imageRows;
    int imageCols;
    int imagePaddedRows;
    int imagePaddedCols;
    int pointSpreadFnRows;
    int pointSpreadFnCols;
    size_t imagePitch;
    size_t imagePaddedPitch;
    size_t pointSpreadFnPitch;

    // device data
    float *d_imageExpected;
    float *d_imageObserved;
    float *d_imageIntrinsic;
    float *d_imageDifferential;
    float *d_pointSpreadFn;
    float *d_pointSpreadFnFlip;

    // execution settings
    dim3 numBlocks;
    dim3 numThreadsPerBlock;
};

#endif