# transformer-cuda-kernel

This project demonstrates the acceleration of a core Transformer component by developing a custom, high-performance CUDA kernel and integrating it into PyTorch as a C++ extension. The focus is on optimizing the softmax function, a primary computational and I/O bottleneck in the self-attention mechanism.

Includes:
- Naive kernel
- Optimized fused kernel using shared memory to perform a parallel reduction
- More optimzied fused kernel using warps and shared memory



Benchmarks

When tensor size becomes (8192, 65536) is when the custom optimized kernel's performance start matching that of PyTorch's Softmax.

