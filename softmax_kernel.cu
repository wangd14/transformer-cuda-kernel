#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void softmax_naive_kernel(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    int row = blockIdx.x + blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    //Pass 1: Max value in row
    float row_max = -FLT_MAX;
    for (int i = 0; i < cols; ++i) {
        row_max = fmaxf(row_max, input[row * cols + i])
    }

    //Pass 2: Sum of Expotentials (normalization factor)
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; ++i) {
        sum_exp += expf(input[row * cols + i] - row_max);
    }
    
    //Pass 3: Normalize and write result to output tensor
    for (int i = 0; i < cols; ++i) {
        output[row * cols + i] = expf(input[row * cols + i] - row_max) / sum_exp;
    }

}

__global__ void softmax_fused_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    //Each block responsible for a row
    //Each thread responsible for a subset of elements in row
    int row = blockIdx.x;
    int tId = threadIdx.x;

    // Shared memory declaration
    // Declared as external array so size can be dynamically set
    extern __shared__ float smem;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = tId; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + i]);
    }

    smem[tId] = local_max;
    // Synchronize to make sure all threads have written their local_max
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tId < stride) {
            smem[tId] = fmaxf(smem[tId], smem[tId + stride]);
        }
        __syncthreads();
    }
    const float row_max = smem;

    for (int i = tId; i < cols; i += blockDim.x) {
        local_sum += expf(input[row * cols + i] - row_max);
    }
    smem[tId] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tId < stride) {
            smem[tId] += smem[tId + stride];
        }
        __syncthreads();
    }
    const float row_sum = smem;

    for (int i = tId; i < cols; i += blockDim.x) {
        output[row * cols + i] = expf(input[row * cols + i] - row_max) / row_sum;
    }
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // 0xFFFFFFFF is warp mask, specifies which threads are needed
        // 32 bits map to 32 threads
        //__shfl_down_sync exchanges data between registers of same warp
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_warp_optimized_kernel(
    const float* input
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tId = threadIdx.x;
    int lane_id = tId % 32; // Thread's index within its warp (0-31)
    int warp_id = tId / 32; // Warp's index within its block (0-15 for a 512-thread block)

    // Shared memory for the final reduction step between warps.
    extern __shared__ float smem;

    float local_max = -FLT_MAX;
    for (int i = tId; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + i]);
    }

    float warp_max = warp_reduce_max(local_max);

    if (lane_id == 0) {
        smem[warp_id] = warp_max;
    }

    __syncthreads();

    float final_max = (tId < (blockDim.x / 32)) ? smem[tId] : -FLT_MAX;
    if (warp_id == 0) {
        final_max = warp_reduce_max(final_max);
    }

    if (tId == 0) {
        smem = final_max;
    }

    __syncthreads();
    const float row_max = smem;

    float local_sum = 0.0f;
    for (int i = tId; i < cols; i += blockDim.x) {
        local_sum += expf(input[row * cols + i] - row_max);
    }

    float warp_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        smem[warp_id] = warp_max;
    }

    __syncthreads();

    float final_sum = (tId < (blockDim.x / 32)) ? smem[tId]: 0;
    if (warp_id == 0) {
        final_sum = warp_reduce_sum(final_sum);
    }

    if (tId == 0) {
        smem = final_sum;
    }

    __syncthreads();
    const float row_sum = smem;

    for (int i =tId; i< cols; i += blockDim.x) {
        output[row * cols + i] = expf(input[row * cols + i] - row_max) / row_sum;
    }

}

void softmax_naive_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output
) {
    //Get dimensions
    int rows = input.size(0);
    int cols = input.size(1);

    //One thread per row
    const int threads_per_block = 256;
    const int num_blocks = (rows + threads_per_block - 1) / threads_per_block;

    softmax_naive_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}

void softmax_fused_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output
) {
    const int rows = input.size(0);
    const int cols = input.size(1);

    //Must be power of 2
    const int threads_per_block = 512;

    const int num_blocks = rows;

    const int shared_mem_size = threads_per_block * sizeof(float);

    softmax_fused_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}

void softmax_warp_optimized_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output
) {
    const int rows = input.size(0);
    const int cols = input.size(1);

    const int threads_per_block = 512;
    const int num_blocks = rows;

    const int num_of_warps = threads_per_block / 32;
    const int shared_mem_size = num_of_warps * sizeof(float);

    softmax_warp_optimized_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
}