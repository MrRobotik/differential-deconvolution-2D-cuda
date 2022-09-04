#ifndef DIFFERENTIALDECONV2D_H
#define DIFFERENTIALDECONV2D_H

#include <iostream>
#include <opencv2/imgproc.hpp>

class DifferentialDeconv2D
{
public:

    DifferentialDeconv2D() = delete;

    DifferentialDeconv2D(
        const cv::Mat &pointSpreadFn,
        double optimizerEta,
        double optimizerLambda,
        unsigned int numIterations,
        unsigned int numThreadsPerBlock);

    cv::Mat operator ()(const cv::Mat &image);

private:

    static void showProgress(int progress);

private:

    cv::Mat pointSpreadFn;
    cv::Mat pointSpreadFnFlip;
    double optimizerEta;
    double optimizerLambda;
    unsigned int numIterations;
    unsigned int numThreadsPerBlock;
};

#endif
