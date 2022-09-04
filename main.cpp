#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "DifferentialDeconv2D.h"

int main(int argc, const char **argv)
{
    if (argc != 4) {
        std::cout << "usage: deconv2d-demo IMAGE PSF ITERATIONS" << std::endl;
        return 1;
    }
    cv::Mat imageOriginal = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat pointSpreadFn = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    unsigned int numIterations = std::strtoul(argv[3], nullptr, 10);
    unsigned int threadsPerBlock = 32;
    double optimizerEta = 0.0125;
    double optimizerLambda = 0.02;
    cv::imshow("original", imageOriginal);

    // convert to floats and normalize
    imageOriginal.convertTo(imageOriginal, CV_32FC1);
    pointSpreadFn.convertTo(pointSpreadFn, CV_32FC1);
    imageOriginal /= 255.0;
    pointSpreadFn /= cv::sum(pointSpreadFn);

    // construct algorithm
    DifferentialDeconv2D deconv2D(
        pointSpreadFn,
        optimizerEta,
        optimizerLambda,
        numIterations,
        threadsPerBlock);

    // deconvolve
    cv::Mat imageIntrinsic = deconv2D(imageOriginal);
    imageIntrinsic.convertTo(imageIntrinsic, CV_8UC1, 255.0);
    cv::imshow("deconvolved", imageIntrinsic);
    cv::waitKey(0);
    return 0;
}
