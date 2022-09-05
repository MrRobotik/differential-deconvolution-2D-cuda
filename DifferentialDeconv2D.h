/*
Deconvolution using gradient descent.
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
