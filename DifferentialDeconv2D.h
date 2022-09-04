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
        unsigned int numIterations,
        unsigned int threadsPerBlock);

    cv::Mat operator ()(const cv::Mat &image);

private:

    cv::Mat pointSpreadFn;
    cv::Mat pointSpreadFnFlip;
    double optimizerEta;
    unsigned int numIterations;
    unsigned int threadsPerBlock;
};

#endif
