#include "DifferentialDeconv2D.h"
#include "cuda_backend/Optimizer.h"

#ifdef DIFFERENTIALDECONV2D_SHOW_PROGRESS
#include <iostream>
#include <iomanip>

static void showProgress(double progress)
{
    std::cout << '\r';
    for (int i = 0; i < int(50.0 * progress); i ++) {
        std::cout << '|';
    }
    for (int i = int(50.0 * progress); i < 50; i ++) {
        std::cout << '=';
    }
    std::cout << std::right << std::setw(4) << int(100.0 * progress);
    std::cout << " %" << std::flush;
}
#endif

DifferentialDeconv2D::DifferentialDeconv2D(
    const cv::Mat &pointSpreadFn,
    double gradientDescentEta,
    double regularizerLambda,
    unsigned int numIterations,
    unsigned int numThreadsPerBlock)
    :
    gradientDescentEta(gradientDescentEta),
    regularizerLambda(regularizerLambda),
    numIterations(numIterations),
    numThreadsPerBlock(numThreadsPerBlock)
{
    if (pointSpreadFn.rows % 2 == 0 || pointSpreadFn.cols % 2 == 0) {
        throw std::logic_error("pointSpreadFn size must be odd");
    }
    pointSpreadFn.convertTo(this->pointSpreadFn, CV_32FC1);
    cv::flip(this->pointSpreadFn, this->pointSpreadFnFlip, -1);
}

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
        cv::BORDER_REFLECT_101);

    // to get the initial observed image
    cv::filter2D(
        imageExpected,
        imageObserved,
        CV_32FC1,
        this->pointSpreadFn,
        cv::Size(-1, -1),
        0.0,
        cv::BORDER_REFLECT_101);

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
        optimizer->step(this->gradientDescentEta, this->regularizerLambda);
        #ifdef DIFFERENTIALDECONV2D_SHOW_PROGRESS
        showProgress(double(i + 1) / double(this->numIterations));
        #endif
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
