#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifndef __CUDACC__
#include <stddef.h>
#endif

class Optimizer
{
public:

    Optimizer() = delete;

    Optimizer(
        unsigned int numThreadsPerBlock,
        const float *h_imageExpected,
        const float *h_imageObserved,
        const float *h_pointSpreadFn,
        const float *h_pointSpreadFnFlip,
        int imageRows,
        int imageCols,
        int imageRowPadding,
        int imageColPadding,
        int pointSpreadFnRows,
        int pointSpreadFnCols);

    ~Optimizer();

    void step(double gradientDescentEta, double regularizerLambda);

    void getResultFromDevice(float *h_imageIntrinsic) const;

private:

    // execution settings
    unsigned int numThreadsPerBlock;

    // device metadata
    int imageRows;
    int imageCols;
    int imageRowPadding;
    int imageColPadding;
    int pointSpreadFnRows;
    int pointSpreadFnCols;
    size_t imagePitch;
    size_t pointSpreadFnPitch;

    // device data
    float *d_imageExpected;
    float *d_imageObserved;
    float *d_imageIntrinsic;
    float *d_imageDifferential;
    float *d_pointSpreadFn;
    float *d_pointSpreadFnFlip;
};

#endif
