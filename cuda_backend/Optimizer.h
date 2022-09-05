/*
Deconvolution using gradient descent (CUDA backend).
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
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifndef __CUDACC__
#include <stddef.h>
#endif

class Optimizer
{
public:

    Optimizer() = delete;

    Optimizer(
        unsigned int numThreadsPerBlock,
        const float *h_imageExpected,
        const float *h_imageObserved,
        const float *h_pointSpreadFn,
        const float *h_pointSpreadFnFlip,
        int imageRows,
        int imageCols,
        int imageRowPadding,
        int imageColPadding,
        int pointSpreadFnRows,
        int pointSpreadFnCols);

    ~Optimizer();

    void step(double gradientDescentEta, double regularizerLambda);

    void getResultFromDevice(float *h_imageIntrinsic) const;

private:

    // execution settings
    unsigned int numThreadsPerBlock;

    // device metadata
    int imageRows;
    int imageCols;
    int imageRowPadding;
    int imageColPadding;
    int pointSpreadFnRows;
    int pointSpreadFnCols;
    size_t imagePitch;
    size_t pointSpreadFnPitch;

    // device data
    float *d_imageExpected;
    float *d_imageObserved;
    float *d_imageIntrinsic;
    float *d_imageDifferential;
    float *d_pointSpreadFn;
    float *d_pointSpreadFnFlip;
};

#endif
