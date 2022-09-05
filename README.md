# differential-deconvolution-2D-cuda

## about

This project is inspired by [1], but instead of random walks optimization, it leverages computational capabilities of modern GPUs and processes all the pixels in parallel at each iteration. Also, instead of uninformed changes of pixel energy by a small amount and checks, whether the objective function is improved or not, the derivative of objective function w.r.t. intrinsic image is computed directly and gradient descent optimization is used. Gamma corrected SSD regularizer is implemented as well. The host code is written in C++ and uses OpenCV library for algorithm initialization, which is not offloaded to GPU. The subsequent iterative gradient descent optimization is performed as a sequence of CUDA kernels in the `for` loop. A demo application is provided in `main.cpp`.

## bibliography

[1] J. Gregson, F. Heide, M. B. Hullin, M. Rouf and W. Heidrich, "Stochastic Deconvolution," 2013 IEEE Conference on Computer Vision and Pattern Recognition, 2013, pp. 1043-1050, doi: 10.1109/CVPR.2013.139.
