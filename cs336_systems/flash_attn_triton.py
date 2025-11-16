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
    FlashAttention-2 前向传播 Triton kernel
    
    参数:
        Q, K, V: 输入张量指针
        O: 输出张量指针
        L: logsumexp 输出指针
        stride_*: 各张量的步长
        N: 序列长度
        d: 特征维度
        BLOCK_M: Q 的块大小 (Br)
        BLOCK_N: K/V 的块大小 (Bc)
        BLOCK_D: 特征维度块大小
    """
    # 获取当前块的索引
    pid_m = tl.program_id(0)
    
    # 计算当前块处理的行范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 创建掩码，确保不越界
    mask_m = offs_m < N
    
    # 计算 Q_i 的指针
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    
    # 加载 Q_i 到 SRAM
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # 初始化输出累加器、统计量
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 内层循环：遍历 K, V 的所有块
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for start_n in range(0, num_blocks_n):
        # 计算当前处理的列范围
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # 加载 K_j 到 SRAM
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        
        # 加载 V_j 到 SRAM
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # 计算注意力分数 S_ij = Q_i @ K_j^T
        s_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s_ij += tl.dot(q, tl.trans(k))
        
        # 创建有效掩码
        mask_ij = mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(mask_ij, s_ij, float("-inf"))
        
        # 计算行最大值: m_ij = max(m_i, rowmax(S_ij))
        m_ij = tl.maximum(m_i, tl.max(s_ij, axis=1))
        
        # 计算注意力权重: P_ij = exp(S_ij - m_ij)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        
        # 更新归一化因子: l_ij = exp(m_i - m_ij) * l_i + rowsum(P_ij)
        alpha = tl.exp(m_i - m_ij)
        l_ij = alpha * l_i + tl.sum(p_ij, axis=1)
        
        # 更新输出累加器
        # O_i = diag(alpha) @ O_i + P_ij @ V_j
        acc = acc * alpha[:, None]
        acc += tl.dot(p_ij.to(v.dtype), v)
        
        # 更新统计量
        m_i = m_ij
        l_i = l_ij
    
    # 最终归一化: O_i = diag(1/l_i) @ O_i
    acc = acc / l_i[:, None]
    
    # 计算 logsumexp: L_i = m_i + log(l_i)
    l_out = m_i + tl.log(l_i)
    
    # 将结果写回 HBM
    o_ptrs = O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None])
    
    # 写回 logsumexp
    l_ptrs = L + offs_m
    tl.store(l_ptrs, l_out, mask=mask_m)


def flash_attn_triton_forward(Q, K, V, block_m=64, block_n=64):
    """
    FlashAttention-2 前向传播的 Triton 实现
    
    参数:
        Q: Query 矩阵，形状 (N, d)
        K: Key 矩阵，形状 (N, d)
        V: Value 矩阵，形状 (N, d)
        block_m: Q 的块大小 (默认 64)
        block_n: K/V 的块大小 (默认 64)
    
    返回:
        O: 输出矩阵，形状 (N, d)
        L: logsumexp，形状 (N,)
    """
    # 检查输入形状
    assert Q.shape == K.shape == V.shape, "Q, K, V 必须有相同的形状"
    N, d = Q.shape
    
    # 确保在同一设备上
    assert Q.device == K.device == V.device, "Q, K, V 必须在同一设备上"
    device = Q.device
    
    # 分配输出张量
    O = torch.empty_like(Q)
    L = torch.empty(N, device=device, dtype=Q.dtype)
    
    # 获取步长
    stride_qm, stride_qd = Q.stride()
    stride_km, stride_kd = K.stride()
    stride_vm, stride_vd = V.stride()
    stride_om, stride_od = O.stride()
    
    # 设置块大小
    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_D = triton.next_power_of_2(d)
    
    # 计算网格大小
    grid = (triton.cdiv(N, BLOCK_M),)
    
    # 启动 kernel
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
    FlashAttention-2 的 PyTorch autograd.Function 封装
    """
    @staticmethod
    def forward(ctx, Q, K, V, block_m=64, block_n=64):
        """
        前向传播
        
        参数:
            Q: Query 矩阵，形状 (N, d)
            K: Key 矩阵，形状 (N, d)
            V: Value 矩阵，形状 (N, d)
            block_m: Q 的块大小
            block_n: K/V 的块大小
        
        返回:
            O: 输出矩阵，形状 (N, d)
        """
        # 确保输入是连续的
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # 调用 Triton kernel
        O, L = flash_attn_triton_forward(Q, K, V, block_m, block_n)
        
        # 保存用于反向传播
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.block_m = block_m
        ctx.block_n = block_n
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("反向传播尚未实现")


def flash_attention_triton(Q, K, V, block_m=64, block_n=64):
    """
    FlashAttention-2 的便捷接口
    
    参数:
        Q: Query 矩阵，形状 (N, d) 或 (batch, N, d)
        K: Key 矩阵，形状 (N, d) 或 (batch, N, d)
        V: Value 矩阵，形状 (N, d) 或 (batch, N, d)
        block_m: Q 的块大小
        block_n: K/V 的块大小
    
    返回:
        O: 输出矩阵，与输入形状相同
    """
    return FlashAttnTriton.apply(Q, K, V, block_m, block_n)


if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(42)
    N, d = 256, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        Q = torch.randn(N, d, device=device, dtype=torch.float32)
        K = torch.randn(N, d, device=device, dtype=torch.float32)
        V = torch.randn(N, d, device=device, dtype=torch.float32)
        
        # Triton 实现
        O_triton = flash_attention_triton(Q, K, V)
        
        # 标准实现（用于验证）
        scores = Q @ K.T
        attn = torch.softmax(scores, dim=-1)
        O_ref = attn @ V
        
        # 比较结果
        print(f"输出形状: {O_triton.shape}")
        print(f"最大差异: {torch.max(torch.abs(O_triton - O_ref)).item():.6f}")
        print(f"相对误差: {torch.mean(torch.abs(O_triton - O_ref) / (torch.abs(O_ref) + 1e-5)).item():.6f}")
    else:
        print("需要 CUDA 设备来运行 Triton kernel")

