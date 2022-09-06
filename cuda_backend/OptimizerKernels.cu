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
#include "OptimizerKernels.h"

__device__ static inline float *elementPtr(
    float *ptr,
    unsigned int row,
    unsigned int col,
    size_t pitch)
{
    char *cptr = reinterpret_cast<char *>(ptr);
    return reinterpret_cast<float *>(cptr + size_t(row) * pitch) + col;
}

__device__ static inline const float *elementPtr(
    const float *ptr,
    unsigned int row,
    unsigned int col,
    size_t pitch)
{
    const char *cptr = reinterpret_cast<const char *>(ptr);
    return reinterpret_cast<const float *>(cptr + size_t(row) * pitch) + col;
}

__device__ static inline const float *nextRowPtr(
    const float *ptr,
    size_t pitch)
{
    const char *cptr = reinterpret_cast<const char *>(ptr);
    return reinterpret_cast<const float *>(cptr + pitch);
}

__global__ void zeroDifferential(
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + imageColPadding;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + imageRowPadding;
    float *dest = elementPtr(d_imageDifferential, row, col, imagePitch);
    *dest = 0.0f;
}

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
    size_t pointSpreadFnPitch)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // initial offset
    d_imageExpected = elementPtr(d_imageExpected, row, col, imagePitch);
    d_imageObserved = elementPtr(d_imageObserved, row, col, imagePitch);

    // differentiate sum of squared differences
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float f_wrt_g = d_imageObserved[j] - d_imageExpected[j];
            float g_wrt_x = d_pointSpreadFnFlip[j];
            accumulator += f_wrt_g * g_wrt_x;
        }
        d_imageExpected = nextRowPtr(d_imageExpected, imagePitch);
        d_imageObserved = nextRowPtr(d_imageObserved, imagePitch);
        d_pointSpreadFnFlip = nextRowPtr(
            d_pointSpreadFnFlip,
            pointSpreadFnPitch);
    }
    // accumulate result
    row += imageRowPadding;
    col += imageColPadding;
    float *dest = elementPtr(d_imageDifferential, row, col, imagePitch);
    *dest += 2.0f * accumulator;
}

__global__ void evalRegularizerDerivative(
    const float *d_imageIntrinsic,
    float *d_imageDifferential,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float regularizerLambda)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + imageColPadding;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + imageRowPadding;

    // read neighbor values
    float central = *elementPtr(d_imageIntrinsic, row, col, imagePitch);
    float xp1 = *elementPtr(d_imageIntrinsic, row, col + 1, imagePitch);
    float xm1 = *elementPtr(d_imageIntrinsic, row, col - 1, imagePitch);
    float yp1 = *elementPtr(d_imageIntrinsic, row + 1, col, imagePitch);
    float ym1 = *elementPtr(d_imageIntrinsic, row - 1, col, imagePitch);

    // differentiate (anisotropic) 2D total variation
    float accumulator =
        ((central - xm1) < 0.0f ? -1.0f : 1.0f) -
        ((xp1 - central) < 0.0f ? -1.0f : 1.0f) +
        ((central - ym1) < 0.0f ? -1.0f : 1.0f) -
        ((yp1 - central) < 0.0f ? -1.0f : 1.0f);

    // accumulate result
    float *dest = elementPtr(d_imageDifferential, row, col, imagePitch);
    *dest += regularizerLambda * accumulator;
}

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
    float gradientDescentEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // initial offset
    d_imageDifferential = elementPtr(d_imageDifferential, row, col, imagePitch);

    // apply convolution change
    float accumulator = 0.0f;
    for (int i = 0; i < pointSpreadFnRows; i ++) {
        for (int j = 0; j < pointSpreadFnCols; j ++) {
            float deltaEnergy = -gradientDescentEta * d_imageDifferential[j];
            accumulator += deltaEnergy * d_pointSpreadFn[j];
        }
        d_imageDifferential = nextRowPtr(d_imageDifferential, imagePitch);
        d_pointSpreadFn = nextRowPtr(d_pointSpreadFn, pointSpreadFnPitch);
    }
    // accumulate result
    row += imageRowPadding;
    col += imageColPadding;
    float *dest = elementPtr(d_imageObserved, row, col, imagePitch);
    *dest += accumulator;
}

__global__ void updateIntrinsic(
    const float *d_imageDifferential,
    float *d_imageIntrinsic,
    int imageRowPadding,
    int imageColPadding,
    size_t imagePitch,
    float gradientDescentEta)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + imageColPadding;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + imageRowPadding;

    // energy change
    d_imageDifferential = elementPtr(d_imageDifferential, row, col, imagePitch);
    float deltaEnergy = -gradientDescentEta * (*d_imageDifferential);
    float *dest = elementPtr(d_imageIntrinsic, row, col, imagePitch);
    *dest += deltaEnergy;
}
