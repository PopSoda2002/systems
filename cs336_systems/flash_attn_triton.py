# FlashAttention-2 Triton Implementation
import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O, L,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    N, d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention-2 forward pass Triton kernel
    
    Args:
        Q, K, V: Input tensor pointers
        O: Output tensor pointer
        L: logsumexp output pointer
        stride_*: Strides for each tensor
        N: Sequence length
        d: Feature dimension
        BLOCK_M: Block size for Q (Br)
        BLOCK_N: Block size for K/V (Bc)
        BLOCK_D: Feature dimension block size
    """
    # Get the current block index
    pid_m = tl.program_id(0)
    
    # Calculate the row range for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Create mask to ensure no out-of-bounds access
    mask_m = offs_m < N
    
    # Calculate pointers for Q_i
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    
    # Load Q_i to SRAM
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize output accumulator and statistics
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Inner loop: iterate over all blocks of K, V
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for start_n in range(0, num_blocks_n):
        # Calculate the column range for the current block
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Load K_j to SRAM
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Load V_j to SRAM
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Compute attention scores S_ij = Q_i @ K_j^T
        s_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s_ij += tl.dot(q, tl.trans(k))
        
        # Create valid mask
        mask_ij = mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(mask_ij, s_ij, float("-inf"))
        
        # Compute row maximum: m_ij = max(m_i, rowmax(S_ij))
        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        
        # Compute attention weights: P_ij = exp(S_ij - m_ij)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        
        # Update normalization factor: l_ij = exp(m_i - m_ij) * l_i + rowsum(P_ij)
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p_ij, axis=1)
        
        # Update output accumulator
        # O_i = diag(alpha) @ O_i + P_ij @ V_j
        acc = acc * alpha[:, None]
        acc += tl.dot(p_ij.to(v.dtype), v)
        
        # Update statistics
        m_i = m_ij
        l_i = l_ij
    
    # Final normalization: O_i = diag(1/l_i) @ O_i
    acc = acc / l_i[:, None]
    
    # Compute logsumexp: L_i = m_i + log(l_i)
    l_out = m_i + tl.log(l_i)
    
    # Write results back to HBM
    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None])
    
    # Write back logsumexp
    l_ptrs = L + offs_m
    tl.store(l_ptrs, l_out, mask=mask_m)


def flash_attn_triton_forward(Q, K, V, block_m=64, block_n=64):
    """
    Triton implementation of FlashAttention-2 forward pass
    
    Args:
        Q: Query matrix, shape (N, d)
        K: Key matrix, shape (N, d)
        V: Value matrix, shape (N, d)
        block_m: Block size for Q (default 64)
        block_n: Block size for K/V (default 64)
    
    Returns:
        O: Output matrix, shape (N, d)
        L: logsumexp, shape (N,)
    """
    # Check input shapes
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
    N, d = Q.shape
    
    # Ensure on the same device
    assert Q.device == K.device == V.device, "Q, K, V must be on the same device"
    device = Q.device
    
    # Allocate output tensors
    O = torch.empty_like(Q)
    L = torch.empty(N, device=device, dtype=Q.dtype)
    
    # Get strides
    stride_qm, stride_qd = Q.stride()
    stride_km, stride_kd = K.stride()
    stride_vm, stride_vd = V.stride()
    stride_om, stride_od = O.stride()
    
    # Set block sizes
    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_D = triton.next_power_of_2(d)
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_M),)
    
    # Launch kernel
    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, L,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        N, d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return O, L


class FlashAttnTriton(torch.autograd.Function):
    """
    PyTorch autograd.Function wrapper for FlashAttention-2
    """
    @staticmethod
    def forward(ctx, Q, K, V, block_m=64, block_n=64):
        """
        Forward pass
        
        Args:
            Q: Query matrix, shape (N, d)
            K: Key matrix, shape (N, d)
            V: Value matrix, shape (N, d)
            block_m: Block size for Q
            block_n: Block size for K/V
        
        Returns:
            O: Output matrix, shape (N, d)
        """
        # Ensure inputs are contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # Call Triton kernel
        O, L = flash_attn_triton_forward(Q, K, V, block_m, block_n)
        
        # Save for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.block_m = block_m
        ctx.block_n = block_n
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass not yet implemented")


def flash_attention_triton(Q, K, V, block_m=64, block_n=64):
    """
    Convenience interface for FlashAttention-2
    
    Args:
        Q: Query matrix, shape (N, d) or (batch, N, d)
        K: Key matrix, shape (N, d) or (batch, N, d)
        V: Value matrix, shape (N, d) or (batch, N, d)
        block_m: Block size for Q
        block_n: Block size for K/V
    
    Returns:
        O: Output matrix, same shape as input
    """
    return FlashAttnTriton.apply(Q, K, V, block_m, block_n)


if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    N, d = 256, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        Q = torch.randn(N, d, device=device, dtype=torch.float32)
        K = torch.randn(N, d, device=device, dtype=torch.float32)
        V = torch.randn(N, d, device=device, dtype=torch.float32)
        
        # Triton implementation
        O_triton = flash_attention_triton(Q, K, V)
        
        # Standard implementation (for validation)
        scores = Q @ K.T
        attn = torch.softmax(scores, dim=-1)
        O_ref = attn @ V
        
        # Compare results
        print(f"Output shape: {O_triton.shape}")
        print(f"Max difference: {torch.max(torch.abs(O_triton - O_ref)).item():.6f}")
        print(f"Relative error: {torch.mean(torch.abs(O_triton - O_ref) / (torch.abs(O_ref) + 1e-5)).item():.6f}")
    else:
        print("CUDA device required to run Triton kernel")

