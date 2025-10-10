from cs336_basics.model import BasicsTransformerLM  # pyright: ignore[reportMissingImports]
import cs336_basics.model

import torch
import timeit
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool
from torch import Tensor
import math
from einops import einsum
from cs336_basics.nn_utils import softmax

@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention with NVTX annotations."""
    
    d_k = K.shape[-1]
    
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return output

def benchmark():
    print("Starting E2E benchmark")
    
    # Replace scaled_dot_product_attention with annotated version
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=1024,
        d_model=512,        
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0,
    ).to("cuda")

    print("Model created")

    input_ids = torch.randint(0, 10000, (1, 1024)).to("cuda")
    num_warmup_steps = 3
    num_steps = 5

    # Benchmark forward pass only
    print("\nBenchmarking forward pass only...")
    model.eval()
    with torch.no_grad():
        # Warm-up phase
        nvtx.range_push("Warmup - Forward Only")
        for _ in range(num_warmup_steps):
            model(input_ids)
        nvtx.range_pop()
        
        # Actual benchmark
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        nvtx.range_push("Benchmark - Forward Only")
        for _ in range(num_steps):
            nvtx.range_push("Forward Pass")
            model(input_ids)
            nvtx.range_pop()
        nvtx.range_pop()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        
    average_time = (end_time - start_time) / num_steps
    print(f"Average time per forward pass: {average_time:.4f} seconds")

    # Benchmark forward + backward pass
    print("\nBenchmarking forward + backward pass...")
    model.train()
    
    # Warm-up phase
    nvtx.range_push("Warmup - Forward + Backward")
    for _ in range(num_warmup_steps):
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        model.zero_grad()
    nvtx.range_pop()
    
    # Actual benchmark
    torch.cuda.synchronize()
    start_time = timeit.default_timer()
    nvtx.range_push("Benchmark - Forward + Backward")
    for _ in range(num_steps):
        nvtx.range_push("Forward Pass")
        logits = model(input_ids)
        loss = logits.sum()
        nvtx.range_pop()
        
        nvtx.range_push("Backward Pass")
        loss.backward()
        nvtx.range_pop()
        
        model.zero_grad()
    nvtx.range_pop()
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    
    average_time = (end_time - start_time) / num_steps
    print(f"Average time per forward + backward pass: {average_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()