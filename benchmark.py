import torch
import time
import softmax

def benchmark(func, input_tensor, warmup_iters=10, timed_iters=100):
    """
    A helper function to benchmark a given function with warmup iterations
    and proper CUDA synchronization.
    """
    # Warmup iterations to handle any initial overheads
    for _ in range(warmup_iters):
        func(input_tensor)
    
    # Synchronize to ensure all warmup kernels are finished
    torch.cuda.synchronize()

    # Timed iterations
    start_time = time.perf_counter()
    for _ in range(timed_iters):
        func(input_tensor)
    
    # Synchronize to ensure all timed kernels are finished
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / timed_iters * 1000
    return avg_time_ms

if __name__ == '__main__':
    # Define a list of tensor sizes to test, including a very large one
    test_sizes = [
        (1024, 8192),
        (2048, 16384),
        (8192, 65536),
        (16384, 65536),
        # (16384, 131072)
    ]

    for size in test_sizes:
        rows, cols = size
        print(f"\n--- Benchmarking on input tensor of size ({rows}, {cols}) ---")
        
        results = {}
        
        try:
            input_tensor = torch.randn(size, device='cuda:0', dtype=torch.float32)

            # --- Run all benchmarks with error handling for OOM ---
            
            # Benchmark PyTorch native
            try:
                results['pytorch'] = benchmark(
                    lambda x: torch.nn.functional.softmax(x, dim=-1), 
                    input_tensor
                )
            except torch.cuda.OutOfMemoryError:
                results['pytorch'] = 'OOM'

            # Benchmark your custom kernels
            # Run naive only for small sizes (skip after rows > 2048)
            if rows <= 2048:
                try:
                    results['naive'] = benchmark(softmax.naive_forward, input_tensor)
                except torch.cuda.OutOfMemoryError:
                    results['naive'] = 'OOM'
            else:
                results['naive'] = 'SKIPPED'

            try:
                results['fused'] = benchmark(softmax.fused_forward, input_tensor)
            except torch.cuda.OutOfMemoryError:
                results['fused'] = 'OOM'
                
            try:
                results['warp'] = benchmark(softmax.warp_optimized_forward, input_tensor)
            except torch.cuda.OutOfMemoryError:
                results['warp'] = 'OOM'

            # --- Print formatted results tables ---

            def time_str(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)

            def speedup_str(base, cur):
                if isinstance(base, float) and isinstance(cur, float) and cur > 0:
                    return f"{base/cur:.2f} x"
                return "N/A"

            # Choose columns depending on whether naive was skipped
            if results.get('naive') == 'SKIPPED':
                headers = ["Implementation", "Time (ms)", "Speedup vs PyTorch"]
                col_widths = [48, 14, 22]
            else:
                headers = ["Implementation", "Time (ms)", "Speedup vs PyTorch", "Speedup vs Naive"]
                col_widths = [42, 14, 22, 22]

            def make_row(items):
                return "| " + " | ".join(f"{str(items[i]):<{col_widths[i]}}" for i in range(len(items))) + " |"

            header = make_row(headers)
            sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
            table_width = len(header)

            print("\n" + "=" * table_width)
            print(header)
            print(sep)

            if results.get('naive') == 'SKIPPED':
                print(make_row(["PyTorch Native Softmax", time_str(results.get('pytorch')), speedup_str(results.get('pytorch'), results.get('pytorch'))]))
                print(make_row(["Custom Naive Kernel", time_str(results.get('naive')), "SKIPPED"]))
                print(make_row(["Custom Fused (Shared Mem) Kernel", time_str(results.get('fused')), speedup_str(results.get('pytorch'), results.get('fused'))]))
                print(make_row(["Custom Fused (Warp Optimized) Kernel", time_str(results.get('warp')), speedup_str(results.get('pytorch'), results.get('warp'))]))
            else:
                print(make_row(["PyTorch Native Softmax", time_str(results.get('pytorch')), speedup_str(results.get('pytorch'), results.get('pytorch')), speedup_str(results.get('naive'), results.get('pytorch'))]))
                print(make_row(["Custom Naive Kernel", time_str(results.get('naive')), speedup_str(results.get('pytorch'), results.get('naive')), speedup_str(results.get('naive'), results.get('naive'))]))
                print(make_row(["Custom Fused (Shared Mem) Kernel", time_str(results.get('fused')), speedup_str(results.get('pytorch'), results.get('fused')), speedup_str(results.get('naive'), results.get('fused'))]))
                print(make_row(["Custom Fused (Warp Optimized) Kernel", time_str(results.get('warp')), speedup_str(results.get('pytorch'), results.get('warp')), speedup_str(results.get('naive'), results.get('warp'))]))

            print("=" * table_width)

            # Clean up memory before the next large allocation
            del input_tensor
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"Could not even create the input tensor of size {size}. Skipping. Your GPU may not have enough memory.")
            continue