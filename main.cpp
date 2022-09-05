#include <iostream>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "DifferentialDeconv2D.h"

int main(int argc, const char **argv)
{
    if (argc != 4) {
        std::cout << "usage: deconv2d-demo IMAGE PSF ITERATIONS" << std::endl;
        return 1;
    }
    std::cout << "loading input data" << std::endl;
    cv::Mat imageOriginal = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat pointSpreadFn = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    unsigned int numIterations = std::strtoul(argv[3], nullptr, 10);
    unsigned int threadsPerBlock = 32u;
    double gradientDescentEta = 0.0125;
    double regularizerLambda = 0.2;
    cv::imshow("original", imageOriginal);

    // convert to floats and normalize
    imageOriginal.convertTo(imageOriginal, CV_32FC1);
    pointSpreadFn.convertTo(pointSpreadFn, CV_32FC1);
    imageOriginal /= 255.0;
    pointSpreadFn /= cv::sum(pointSpreadFn);

    // construct algorithm
    DifferentialDeconv2D deconv2D(
        pointSpreadFn,
        gradientDescentEta,
        regularizerLambda,
        numIterations,
        threadsPerBlock);

    // deconvolve
    std::cout << "running optimization on CUDA device" << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat imageIntrinsic = deconv2D(imageOriginal);
    size_t msDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "\noptimization took " << msDuration << " ms\n" << std::endl;
    imageIntrinsic.convertTo(imageIntrinsic, CV_8UC1, 255.0);
    cv::imshow("deconvolved", imageIntrinsic);
    cv::waitKey(0);
    return 0;
}
