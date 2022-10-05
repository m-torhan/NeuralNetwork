# NeuralNetwork
c++/asm implementation of neural network.

The layers works on type `Tensor` which represents $n$-dimensional array and supports mathematical operations (addition, subtraction, dot product, tensor product, $\ldots$) and other not math realted (adding padding, shuffling, reshaping $\ldots$). Some of the operations are optimized using AVX128 instructions, which made them a lot faster.

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

Build scripts:
 - [`Build/build_debug.sh`](./Build/build_debug.sh)/[`Build/build_debug.bat`](Build/build_debug.bat) - builds debug configuration,
 - [`Build/build_release.sh`](Build/build_release.sh)/[`Build/build_release.bat`](Build/build_release.bat) - builds release configuration,
 - [`Build/build_release_sse.sh`](Build/build_release_sse.sh)/[`Build/build_release_sse.bat`](Build/build_release_sse.bat) - builds release configuration using assembly optimizations,
 - [`Build/clean.sh`](Build/clean.sh)/[`Build/clean.bat`](Build/clean.bat) - cleans all builds.
  
Other scripts:
 - [`Build/generic_builder.sh`](Build/generic_builder.sh)/[`Build/generic_builder.bat`](Build/generic_builder.bat) - generic builder script used in all build scripts,
 - [`Build/unit_tests_run.sh`](Build/unit_tests_run.sh)/[`Build/unit_tests_run.bat`](Build/unit_tests_run.bat) - runs unit tests,
 - [`Build/performance_tests_run.sh`](Build/performance_tests_run.sh) - runs performance tests (available only on Linux),
 - [`Build/generate_performance_report.py`](Build/generate_performance_report.py) - runs performance tests, saves the results and saves them on plots ($y$ axis is the measured time and $x$ axis is commit hash). Results can be found here: [`Build/performance_report/repord.md`](Build/performance_report/report.md).