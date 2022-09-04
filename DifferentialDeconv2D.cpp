#include "DifferentialDeconv2D.h"
#include "cuda_backend/Optimizer.h"

DifferentialDeconv2D::DifferentialDeconv2D(
    const cv::Mat &pointSpreadFn,
    double optimizerEta,
    double optimizerLambda,
    unsigned int numIterations,
    unsigned int threadsPerBlock)
    :
    optimizerEta(optimizerEta),
    optimizerLambda(optimizerLambda),
    numIterations(numIterations),
    threadsPerBlock(threadsPerBlock)
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

    // for efficient extrapolation
    int rowPadding = this->pointSpreadFn.rows / 2;
    int colPadding = this->pointSpreadFn.cols / 2;
    cv::copyMakeBorder(
        image,
        imageExpected,
        rowPadding,
        rowPadding,
        colPadding,
        colPadding,
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
        imageExpected.ptr<const float>(),
        imageObserved.ptr<const float>(),
        this->pointSpreadFn.ptr<const float>(),
        this->pointSpreadFnFlip.ptr<const float>(),
        image.rows,
        image.cols,
        this->pointSpreadFn.rows,
        this->pointSpreadFn.cols);

    // iterative optimization
    for (size_t i = 0; i < this->numIterations; i ++) {
        optimizer->step(this->optimizerEta, this->optimizerLambda);
        int progress = 50.0 * (double(i) / double(this->numIterations));
        showProgress(progress);
        std::cout << std::flush;
    }
    showProgress(50);
    std::cout << std::endl;

    // finalize
    cv::Mat imageIntrinsic(imageExpected.size(), CV_32FC1);
    optimizer->getResultFromDevice(imageIntrinsic.ptr<float>());
    delete optimizer;
    cv::Range rowRange(rowPadding, rowPadding + image.rows);
    cv::Range colRange(colPadding, colPadding + image.cols);
    return imageIntrinsic(rowRange, colRange).clone();
}

void DifferentialDeconv2D::showProgress(int progress)
{
    std::cout << '\r';
    for (int i = 0; i < progress; i ++) {
        std::cout << '|';
    }
    for (int i = progress; i < 50; i ++) {
        std::cout << '=';
    }
}
