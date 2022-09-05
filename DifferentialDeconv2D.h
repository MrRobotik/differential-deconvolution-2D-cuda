#ifndef DIFFERENTIALDECONV2D_H
#define DIFFERENTIALDECONV2D_H

#include <opencv2/imgproc.hpp>

class DifferentialDeconv2D
{
public:

    DifferentialDeconv2D() = delete;

    DifferentialDeconv2D(
        const cv::Mat &pointSpreadFn,
        double gradientDescentEta,
        double regularizerLambda,
        unsigned int numIterations,
        unsigned int numThreadsPerBlock);

    cv::Mat operator ()(const cv::Mat &image);

private:

    static int getAlignment(int n, int blockSize);

private:

    // point-spread-function
    cv::Mat pointSpreadFn;
    cv::Mat pointSpreadFnFlip;

    // settings
    double gradientDescentEta;
    double regularizerLambda;
    unsigned int numIterations;
    unsigned int numThreadsPerBlock;
};

#endif
