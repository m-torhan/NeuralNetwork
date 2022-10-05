# NeuralNetwork
c++/asm implementation of neural network.

Example uses of `NeuralNetwork` class can be found in:
 -  [applications/mnist](./applications/mnist/) digit recognition,
 -  [applications/mnist_vae](./applications/mnist_vae/) digit generation.

At this moment there are implemented layers of type:
 - dense,
 - activation (sigmoid, ReLU, leakyReLU),
 - conv2D (not optimized yet),
 - pool2D (max or mean pooling),
 - reshape,
 - flatten,
 - dropout,
 - normal distribution layer (for VAE).

The layers works on type `Tensor` which represents $n$-dimensional array and supports mathematical operations (addition, subtraction, dot product, tensor product, $\ldots$) and other not math realted (adding padding, shuffling, reshaping $\ldots$). Some of the operations are optimized using AVX128 instructions, which made them a lot faster.