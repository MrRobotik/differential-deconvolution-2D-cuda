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
#ifndef OPTIMIZERKERNELS_H
#define OPTIMIZERKERNELS_H
#ifdef __CUDACC__

__global__ void zeroDifferential(
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch);

__global__ void evalObjectiveFnDerivative(
    const float *d_imageExpected,
    const float *d_imageObserved,
    const float *d_pointSpreadFnFlip,
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t pointSpreadFnPitch);

__global__ void evalRegularizerDerivative(
    const float *d_imageIntrinsic,
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float regularizerLambda);

__global__ void updateObserved(
    const float *d_imageDifferential,
    const float *d_pointSpreadFn,
    float *d_imageObserved,
    int imageRowPadding,
    int imageColPadding,
    int pointSpreadFnRows,
    int pointSpreadFnCols,
    size_t imagePitch,
    size_t pointSpreadFnPitch,
    float gradientDescentEta);

__global__ void updateIntrinsic(
    const float *d_imageDifferential,
    float *d_imageIntrinsic,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float gradientDescentEta);

#endif
#endif
