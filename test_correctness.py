import torch
import softmax # This is your compiled C++/CUDA module

def test_correctness():
    print("--- Running Correctness Test ---")
    
    try:
        # Using a non-power-of-2 size to make it a robust test
        input_tensor = torch.randn(
            (128, 4099), 
            device='cuda:0', 
            dtype=torch.float32
        )
    except Exception as e:
        print(f"Could not create tensor on GPU. Please ensure CUDA is available. Error: {e}")
        return

    # 1. Get the expected "ground truth" output from PyTorch's native softmax
    expected_output = torch.nn.functional.softmax(input_tensor, dim=-1)

    # 2. Get the output from your custom kernels
    print("Running custom kernels...")
    naive_output = softmax.naive_forward(input_tensor)
    fused_output = softmax.fused_forward(input_tensor)
    warp_optimized_output = softmax.warp_optimized_forward(input_tensor)

    # 3. Compare the results
    # For the naive kernel, we use a slightly higher tolerance to account for
    # the different (but still valid) order of floating-point operations.
    naive_correct = torch.allclose(naive_output, expected_output, rtol=1e-05, atol=1e-07)

    # The fused kernels should match the parallel implementation of PyTorch more closely.
    fused_correct = torch.allclose(fused_output, expected_output)
    warp_correct = torch.allclose(warp_optimized_output, expected_output)

    print(f"\nResults:")
    print(f"  Naive Kernel Correct:           {naive_correct}")
    print(f"  Fused (Shared Mem) Correct:     {fused_correct}")
    print(f"  Warp Optimized Kernel Correct:  {warp_correct}")
    print("-" * 30)

if __name__ == '__main__':
    test_correctness()