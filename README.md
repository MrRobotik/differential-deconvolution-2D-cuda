# differential-deconvolution-2D-cuda

# theory

This project is inspired by [1], but instead of random walks optimization, it leverages computational capabilities of modern GPUs and processes all pixels in parallel at each iteration. Also, instead of uninformed changes of pixel energy by a small amount and checking, whether the objective function is improved or not, the derivative of objective function w.r.t. intrinsic image is computed directly and gradient descent optimization (parametrized by rate $ \eta $) is used instead. For the SSD objective, the following expression is used, where $ F $ is intrinsic image, $ G $ is expected image, $ K $ is point-spread-function, and $ K^\prime $ is $ K $ flipped in both axes:
$$
\begin{equation}
\frac{\partial L}{\partial F_{i,j}} = 2 \cdot \sum_{n,m}{\left(\left( F \ast K \right)_{i+n,j+m} - G_{i+n,j+m} \right)} \cdot K_{n,m}^\prime
\end{equation}
$$

Gamma corrected SSD regularizer is implemented as well (weighted by $ \lambda $ parameter). For this, the derivative formula is:
$$
\begin{equation}
\frac{\partial L}{\partial F_{i,j}} = \sum{n,m}{\left( 1 - \frac{F_{i+n,j+m}}{\sqrt{F_{i,j} \cdot F_{i+n,j+m}}} \right)}
\end{equation}
$$

# implementation

The host code is written in C++ and uses OpenCV library for algorithm initialization, which is not offloaded to GPU. The following iterative gradient descent optimization is performed as a sequence of CUDA kernels in the `for` loop. A demo application is provided in `main.cpp`.

---

[1] J. Gregson, F. Heide, M. B. Hullin, M. Rouf and W. Heidrich, "Stochastic Deconvolution," 2013 IEEE Conference on Computer Vision and Pattern Recognition, 2013, pp. 1043-1050, doi: 10.1109/CVPR.2013.139.