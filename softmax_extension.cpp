#include <torch/extension.h>
#include <vector>

// Cuda Forward declaration
void softmax_naive_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output);

void softmax_fused_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output);

void softmax_warp_optimized_cuda_launcher(
    torch::Tensor input,
    torch::Tensor output);

torch::Tensor softmax_naive_forward(torch::Tensor input) {
    //Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA tensor");
    TORCH_CHECK(input.is_continguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto output = torch::empty_like(input);

    softmax_naive_cuda_launcher(input, output);
    
    return output;
}

torch::Tensor softmax_fused_forward(torch::Tensor input) {
    //Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA tensor");
    TORCH_CHECK(input.is_continguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto output = torch::empty_like(input);

    softmax_fused_cuda_launcher(input, output);
    
    return output;
}

torch::Tensor softmax_warp_optimized_foward(torch::Tensor input) {
    //Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA tensor");
    TORCH_CHECK(input.is_continguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be of type float32");

    auto output = torch::empty_like(input);

    softmax_warp_optimized_cuda_launcher(input, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_forward", &softmax_naive_forward, "Softmax forward pass (naive CUDA)");
    m.def("fused_forward", &softmax_fused_forward, "Softmax forward pass (fused CUDA)");
    m.def("warp_optimized_forward", &softmax_warp_optimized_foward, "Softmax forward pass (fused CUDA with warp primitives)");
}