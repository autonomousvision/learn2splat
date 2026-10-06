import gc
import torch
from torch.autograd import profiler


def report_gpu_tensors(device='cuda:0'):
    tensors_info = []
    total = 0

    # Scan all objects tracked by Python
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and str(obj.device) == device:
                # Calculate memory
                mem = obj.numel() * obj.element_size() / 1024**2
                total += mem

                # Try to find variable names pointing to this tensor
                names = [name for name, val in globals().items() if val is obj]
                name_str = ", ".join(names) if names else "<anonymous>"

                tensors_info.append((name_str, tuple(obj.shape), str(obj.dtype), mem))
        except:
            pass

    # Sort by memory usage descending
    tensors_info.sort(key=lambda x: -x[3])

    # Print nicely
    print(f"{'Name(s)':>30} | {'Shape':>20} | {'Dtype':>10} | {'Memory (MB)':>12}")
    print("-" * 80)
    for name, shape, dtype, mem in tensors_info:
        print(f"{name:>30} | {str(shape):>20} | {dtype:>10} | {mem:12.2f}")
    print("-" * 80)
    print(f"Total tracked GPU tensor memory: {total:.2f} MB")


def profile_gpu_memory(fn, *args, top_n=10, **kwargs):
    """
    Profile GPU memory usage including custom CUDA kernels.
    Prints peak memory and top PyTorch operations.

    Args:
        fn: function to call (e.g., model.forward)
        *args, **kwargs: arguments to pass to fn
        top_n: number of top memory-consuming operations to print
    """
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Record memory before call
    mem_before = torch.cuda.memory_allocated()

    # Use PyTorch profiler for visible PyTorch ops
    with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()

    # Record memory after call
    mem_after = torch.cuda.memory_allocated()
    mem_diff = (mem_after - mem_before) / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print("\n=== GPU Memory Profiling ===")
    print("Function:", fn.__name__)
    print(f"\nMemory before call: {mem_before / 1024**3:.2f} GiB")
    print(f"Memory after call : {mem_after / 1024**3:.2f} GiB")
    print(f"Memory diff       : {mem_diff:.2f} GiB")
    print(f"Peak allocated    : {peak:.2f} GiB\n")

    # Get key averages from profiler
    key_avg = prof.key_averages()
    key_avg_sorted = sorted(
        key_avg,
        key=lambda k: getattr(k, "self_cuda_memory_usage", 0),
        reverse=True
    )

    # Print top N operations
    print(f"{'Operation':<40} | {'CUDA Memory (MB)':>15} | {'Shape info':>20} | #Calls")
    print("-" * 100)
    for evt in key_avg_sorted[:top_n]:
        mem_mb = getattr(evt, "self_cuda_memory_usage", 0) / 1024**2
        shapes = str(evt.input_shapes) if hasattr(evt, 'input_shapes') else "-"
        print(f"{evt.key:<40} | {mem_mb:15.2f} | {shapes:>20} | {evt.count}")

    return result