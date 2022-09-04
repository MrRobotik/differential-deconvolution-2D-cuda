#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "DifferentialDeconv2D.h"

int main(int argc, const char **argv)
{
    if (argc != 2) {
        std::cout << "usage: deconv2d-demo IMAGE PSF" << std::endl;
        return 1;
    }
    cv::Mat kernel = cv::getGaussianKernel(15, -1.0);
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat pointSpreadFn = kernel * kernel.t();
    pointSpreadFn /= cv::sum(pointSpreadFn);
    cv::filter2D(image, image, CV_8UC1, pointSpreadFn);
    cv::imshow("input", image);
    image.convertTo(image, CV_32FC1, 1.0 / 255.0);
    DifferentialDeconv2D deconv2D(pointSpreadFn, 0.05, 500u, 64u);
    cv::Mat imageIntrinsic = deconv2D(image);
    imageIntrinsic.convertTo(imageIntrinsic, CV_8UC1, 255.0);
    cv::imshow("output", imageIntrinsic);
    cv::waitKey(0);
    return 0;
}
