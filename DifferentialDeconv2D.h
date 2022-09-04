#ifndef DIFFERENTIALDECONV2D_H
#define DIFFERENTIALDECONV2D_H

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
        unsigned int numThreadsPerBlock,
        void (*afterIterationCallback)(double)=nullptr);

    cv::Mat operator ()(const cv::Mat &image);

private:

    cv::Mat pointSpreadFn;
    cv::Mat pointSpreadFnFlip;
    double optimizerEta;
    double optimizerLambda;
    unsigned int numIterations;
    unsigned int numThreadsPerBlock;
    void (*afterIterationCallback)(double);
};

#endif
