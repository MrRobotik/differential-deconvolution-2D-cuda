#include "DifferentialDeconv2D.h"
#include "cuda_backend/Optimizer.h"

DifferentialDeconv2D::DifferentialDeconv2D(
    const cv::Mat &pointSpreadFn,
    double optimizerEta,
    double optimizerLambda,
    unsigned int numIterations,
    unsigned int numThreadsPerBlock,
    void (*postIterationCallback)(double))
    :
    optimizerEta(optimizerEta),
    optimizerLambda(optimizerLambda),
    numIterations(numIterations),
    numThreadsPerBlock(numThreadsPerBlock),
    postIterationCallback(postIterationCallback)
{
    if (pointSpreadFn.rows % 2 == 0 || pointSpreadFn.cols % 2 == 0) {
        throw std::logic_error("pointSpreadFn size must be odd");
    }
    pointSpreadFn.convertTo(this->pointSpreadFn, CV_32FC1);
    cv::flip(this->pointSpreadFn, this->pointSpreadFnFlip, -1);
}
#include <iostream>
cv::Mat DifferentialDeconv2D::operator ()(const cv::Mat &image)
{
    if (image.type() != CV_32FC1) {
        throw std::logic_error("image type must be CV_32FC1");
    }
    cv::Mat imageExpected;
    cv::Mat imageObserved;

    // for efficient extrapolation & device alignment
    int rowPadding = this->pointSpreadFn.rows / 2;
    int colPadding = this->pointSpreadFn.cols / 2;
    int rowAlignment = getAlignment(image.rows, this->numThreadsPerBlock);
    int colAlignment = getAlignment(image.cols, this->numThreadsPerBlock);
    cv::copyMakeBorder(
        image,
        imageExpected,
        rowPadding,
        rowPadding + (rowAlignment - image.rows),
        colPadding,
        colPadding + (colAlignment - image.cols),
        cv::BORDER_REPLICATE);

    // to get the initial observed image
    cv::filter2D(
        imageExpected,
        imageObserved,
        CV_32FC1,
        this->pointSpreadFn,
        cv::Size(-1, -1),
        0.0,
        cv::BORDER_REPLICATE);

    // construct optimizer
    Optimizer *optimizer = new Optimizer(
        this->numThreadsPerBlock,
        imageExpected.ptr<const float>(),
        imageObserved.ptr<const float>(),
        this->pointSpreadFn.ptr<const float>(),
        this->pointSpreadFnFlip.ptr<const float>(),
        imageExpected.rows,
        imageExpected.cols,
        rowPadding,
        colPadding,
        this->pointSpreadFn.rows,
        this->pointSpreadFn.cols);

    // iterative optimization
    for (size_t i = 0; i < this->numIterations; i ++) {
        optimizer->step(this->optimizerEta, this->optimizerLambda);
        if (this->postIterationCallback != nullptr) {
            double progress = double(i + 1) / double(this->numIterations);
            this->postIterationCallback(progress);
        }
    }
    // finalize
    cv::Mat imageIntrinsic(imageExpected.size(), CV_32FC1);
    optimizer->getResultFromDevice(imageIntrinsic.ptr<float>());
    delete optimizer;
    cv::Range rowRange(rowPadding, rowPadding + image.rows);
    cv::Range colRange(colPadding, colPadding + image.cols);
    return imageIntrinsic(rowRange, colRange).clone();
}

int DifferentialDeconv2D::getAlignment(int n, int blockSize)
{
    return ((n + (blockSize - 1)) / blockSize) * blockSize;
}
