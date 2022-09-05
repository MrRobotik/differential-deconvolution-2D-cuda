#include "Optimizer.h"
#include "OptimizerKernels.h"

Optimizer::Optimizer(
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
    int pointSpreadFnCols)
    :
    numThreadsPerBlock(numThreadsPerBlock),
    imageRows(imageRows),
    imageCols(imageCols),
    imageRowPadding(imageRowPadding),
    imageColPadding(imageColPadding),
    pointSpreadFnRows(pointSpreadFnRows),
    pointSpreadFnCols(pointSpreadFnCols)
{
    // allocate resources on device
    cudaMallocPitch(
        &(this->d_imageExpected),
        &(this->imagePitch),
        this->imageCols * sizeof(float),
        this->imageRows);
    cudaMallocPitch(
        &(this->d_imageObserved),
        &(this->imagePitch),
        this->imageCols * sizeof(float),
        this->imageRows);
    cudaMallocPitch(
        &(this->d_imageIntrinsic),
        &(this->imagePitch),
        this->imageCols * sizeof(float),
        this->imageRows);
    cudaMallocPitch(
        &(this->d_imageDifferential),
        &(this->imagePitch),
        this->imageCols * sizeof(float),
        this->imageRows);
    cudaMallocPitch(
        &(this->d_pointSpreadFn),
        &(this->pointSpreadFnPitch),
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnRows);
    cudaMallocPitch(
        &(this->d_pointSpreadFnFlip),
        &(this->pointSpreadFnPitch),
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnRows);

    // copy input data
    cudaMemcpy2D(
        this->d_imageExpected,
        this->imagePitch,
        h_imageExpected,
        this->imageCols * sizeof(float),
        this->imageCols * sizeof(float),
        this->imageRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_imageObserved,
        this->imagePitch,
        h_imageObserved,
        this->imageCols * sizeof(float),
        this->imageCols * sizeof(float),
        this->imageRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_imageIntrinsic,
        this->imagePitch,
        h_imageExpected,
        this->imageCols * sizeof(float),
        this->imageCols * sizeof(float),
        this->imageRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_pointSpreadFn,
        this->pointSpreadFnPitch,
        h_pointSpreadFn,
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_pointSpreadFnFlip,
        this->pointSpreadFnPitch,
        h_pointSpreadFnFlip,
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnCols * sizeof(float),
        this->pointSpreadFnRows,
        cudaMemcpyHostToDevice);

    // initialize differential image to zeros
    size_t total = size_t(this->imageRows) * size_t(this->imageCols);
    float *zeros = new float[total];
    for (size_t i = 0; i < total; i ++) {
        zeros[i] = 0.0f;
    }
    cudaMemcpy2D(
        this->d_imageDifferential,
        this->imagePitch,
        zeros,
        this->imageCols * sizeof(float),
        this->imageCols * sizeof(float),
        this->imageRows,
        cudaMemcpyHostToDevice);
    delete [] zeros;
}

Optimizer::~Optimizer()
{
    // free all device resources
    cudaFree(this->d_imageExpected);
    cudaFree(this->d_imageObserved);
    cudaFree(this->d_imageIntrinsic);
    cudaFree(this->d_imageDifferential);
    cudaFree(this->d_pointSpreadFn);
    cudaFree(this->d_pointSpreadFnFlip);
}

void Optimizer::step(double gradientDescentEta, double regularizerLambda)
{
    unsigned int rows = this->imageRows - 2 * this->imageRowPadding;
    unsigned int cols = this->imageCols - 2 * this->imageColPadding;
    dim3 blockDim(this->numThreadsPerBlock, this->numThreadsPerBlock);
    dim3 gridDim(cols / blockDim.x, rows / blockDim.y);

    evalObjectiveFnDerivative<<<gridDim, blockDim>>>(
        this->d_imageExpected,
        this->d_imageObserved,
        this->d_pointSpreadFnFlip,
        this->d_imageDifferential,
        this->imageRowPadding,
        this->imageColPadding,
        this->pointSpreadFnRows,
        this->pointSpreadFnCols,
        this->imagePitch,
        this->pointSpreadFnPitch);

    if (regularizerLambda != 0.0) {
        evalRegularizerDerivative<<<gridDim, blockDim>>>(
            this->d_imageIntrinsic,
            this->d_imageDifferential,
            this->imageRowPadding,
            this->imageColPadding,
            this->imagePitch,
            regularizerLambda);
    } // else don't use regularizer

    updateObserved<<<gridDim, blockDim>>>(
        this->d_imageDifferential,
        this->d_pointSpreadFn,
        this->d_imageObserved,
        this->imageRowPadding,
        this->imageColPadding,
        this->pointSpreadFnRows,
        this->pointSpreadFnCols,
        this->imagePitch,
        this->pointSpreadFnPitch,
        gradientDescentEta);

    updateIntrinsic<<<gridDim, blockDim>>>(
        this->d_imageDifferential,
        this->d_imageIntrinsic,
        this->imageRowPadding,
        this->imageColPadding,
        this->imagePitch,
        gradientDescentEta);

    zeroDifferential<<<gridDim, blockDim>>>(
        this->d_imageDifferential,
        this->imageRowPadding,
        this->imageColPadding,
        this->imagePitch);
}

void Optimizer::getResultFromDevice(float *h_imageIntrinsic) const
{
    cudaMemcpy2D(
        h_imageIntrinsic,
        this->imageCols * sizeof(float),
        this->d_imageIntrinsic,
        this->imagePitch,
        this->imageCols * sizeof(float),
        this->imageRows,
        cudaMemcpyDeviceToHost);
}
