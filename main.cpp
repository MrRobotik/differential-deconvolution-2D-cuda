/*
Demo application for DifferentialDeconv2D algorithm.
Copyright (C) 2022  Ing. Adam Kucera

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "DifferentialDeconv2D.h"

int main(int argc, const char **argv)
{
    if (argc != 4) {
        std::cout << "usage: deconv2D-demo IMAGE PSF ITERATIONS\n";
        std::cout << "  IMAGE       to be deconvolved\n";
        std::cout << "  PSF         point-spread-function\n";
        std::cout << "  ITERATIONS  number of iterations\n";
        std::cout << std::endl;
        return 1;
    }
    std::cout << "loading input data" << std::endl;
    cv::Mat imageOriginal = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat pointSpreadFn = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    unsigned int numIterations = std::strtoul(argv[3], nullptr, 10);
    unsigned int threadsPerBlock = 32u;
    double gradientDescentEta = 0.01;
    double regularizerLambda = 0.005;
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
