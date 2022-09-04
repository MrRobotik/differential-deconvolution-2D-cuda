#include "Optimizer.h"
#include "OptimizerKernels.h"

Optimizer::Optimizer(
    const float *h_imageExpected,
    const float *h_imageObserved,
    const float *h_pointSpreadFn,
    const float *h_pointSpreadFnFlip,
    int imageRows,
    int imageCols,
    int pointSpreadFnRows,
    int pointSpreadFnCols)
    :
    imageRows(imageRows),
    imageCols(imageCols),
    imagePaddedRows(imageRows + (pointSpreadFnRows - 1)),
    imagePaddedCols(imageCols + (pointSpreadFnCols - 1)),
    pointSpreadFnRows(pointSpreadFnRows),
    pointSpreadFnCols(pointSpreadFnCols)
{
    // allocate resources on device
    cudaMallocPitch(
        &(this->d_imageExpected),
        &(this->imagePaddedPitch),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows);
    cudaMallocPitch(
        &(this->d_imageObserved),
        &(this->imagePaddedPitch),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows);
    cudaMallocPitch(
        &(this->d_imageIntrinsic),
        &(this->imagePaddedPitch),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows);
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
        this->imagePaddedPitch,
        h_imageExpected,
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_imageObserved,
        this->imagePaddedPitch,
        h_imageObserved,
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows,
        cudaMemcpyHostToDevice);
    cudaMemcpy2D(
        this->d_imageIntrinsic,
        this->imagePaddedPitch,
        h_imageExpected,
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows,
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

    // prepare execution settings
    this->numThreadsPerBlock.x = 32u;
    this->numThreadsPerBlock.y = 32u;
    this->numBlocks.x = this->imageCols / this->numThreadsPerBlock.x;
    this->numBlocks.y = this->imageRows / this->numThreadsPerBlock.y;
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

void Optimizer::step(double optimizerEta)
{
    dim3 nb = this->numBlocks;
    dim3 nt = this->numThreadsPerBlock;

    zeroDifferential<<<nb, nt>>>(
        this->d_imageDifferential,
        this->imagePitch);

    evalObjectiveFnDerivate<<<nb, nt>>>(
        this->d_imageExpected,
        this->d_imageObserved,
        this->d_pointSpreadFnFlip,
        this->d_imageDifferential,
        this->pointSpreadFnRows,
        this->pointSpreadFnCols,
        this->imagePitch,
        this->imagePaddedPitch,
        this->pointSpreadFnPitch);

    updateObserved<<<nb, nt>>>(
        this->d_imageDifferential,
        this->d_pointSpreadFn,
        this->d_imageObserved,
        this->pointSpreadFnRows,
        this->pointSpreadFnCols,
        this->imagePitch,
        this->imagePaddedPitch,
        this->pointSpreadFnPitch,
        optimizerEta);

    updateIntrinsic<<<nb, nt>>>(
        this->d_imageDifferential,
        this->d_imageIntrinsic,
        this->pointSpreadFnRows,
        this->pointSpreadFnCols,
        this->imagePitch,
        this->imagePaddedPitch,
        optimizerEta);
}

void Optimizer::getResultFromDevice(float *h_imageIntrinsic) const
{
    cudaMemcpy2D(
        h_imageIntrinsic,
        this->imagePaddedCols * sizeof(float),
        this->d_imageIntrinsic,
        this->imagePaddedPitch,
        this->imagePaddedCols * sizeof(float),
        this->imagePaddedRows,
        cudaMemcpyDeviceToHost);
}
