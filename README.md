# ML

CUDA-Accelerated Neural Network from Scratch in python

A high-performance implementation of a fully-connected neural network using CUDA and the CuPy library. The network features two hidden layers with ReLU activation and includes gradient clipping for stability.

Key Features:

Architecture: 4-layer fully-connected network (input, two hidden, output)

Activation: ReLU non-linearity in hidden layers

Optimization: Gradient clipping for stability

Performance: GPU-accelerated via CUDA and CuPy library

Validation: Includes MNIST handwritten digit classification example

Test Command:
```bash
python test_gpu.py
python test_cpu.py
```