# Flash Attn 2
import torch
import math

class FlashAttnTorch(torch.autograd.Function):
    @staticmethod
    # Q: N x d, K: N x d, V: N x d
    def forward(ctx, Q, K, V, Br=None, Bc=None):
        """
        FlashAttention-2 前向传播
        
        参数:
            Q: Query矩阵，形状 (N, d)
            K: Key矩阵，形状 (N, d)
            V: Value矩阵，形状 (N, d)
            Br: Query的块大小 (默认为 N//8)
            Bc: Key/Value的块大小 (默认为 N//4)
        
        返回:
            O: 输出矩阵，形状 (N, d)
            L: logsumexp，形状 (N,)
        """
        N, d = Q.shape
        device = Q.device
        
        # 设置默认块大小
        if Br is None:
            Br = max(1, N // 8)
        if Bc is None:
            Bc = max(1, N // 4)
        
        # 计算块数量
        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)
        
        # 初始化输出矩阵 O 和 logsumexp L
        O = torch.zeros_like(Q)
        L = torch.zeros(N, device=device, dtype=Q.dtype)
        
        # 外层循环：遍历 Q 的每个块
        for i in range(Tr):
            # 步骤 1: 从 HBM 加载 Q_i 到 SRAM
            start_i = i * Br
            end_i = min((i + 1) * Br, N)
            Q_i = Q[start_i:end_i, :]  # 形状: (Br, d)
            
            # 步骤 2: 在芯片上初始化
            O_i = torch.zeros((end_i - start_i, d), device=device, dtype=Q.dtype)
            l_i = torch.zeros(end_i - start_i, device=device, dtype=Q.dtype)
            m_i = torch.full((end_i - start_i,), float('-inf'), device=device, dtype=Q.dtype)
            
            # 内层循环：遍历 K, V 的每个块
            for j in range(Tc):
                # 步骤 3: 从 HBM 加载 K_j, V_j 到 SRAM
                start_j = j * Bc
                end_j = min((j + 1) * Bc, N)
                K_j = K[start_j:end_j, :]  # 形状: (Bc, d)
                V_j = V[start_j:end_j, :]  # 形状: (Bc, d)
                
                # 步骤 4: 在芯片上计算 S_i^(j) = Q_i K_j^T
                S_ij = torch.matmul(Q_i, K_j.T)  # 形状: (Br, Bc)
                
                # 步骤 5: 在芯片上计算统计量和注意力权重
                # m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
                m_ij = torch.maximum(m_i, torch.max(S_ij, dim=1).values)
                
                # P_i^(j) = exp(S_i^(j) - m_i^(j))
                P_ij = torch.exp(S_ij - m_ij.unsqueeze(1))
                
                # l_i^(j) = e^(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P_i^(j))
                l_ij = torch.exp(m_i - m_ij) * l_i + torch.sum(P_ij, dim=1)
                
                # 步骤 6: 在芯片上更新输出
                # O_i^(j) = diag(e^(m_i^(j-1) - m_i^(j)))^(-1) * O_i^(j-1) + P_i^(j) V_j
                O_i = torch.diag(torch.exp(m_i - m_ij)) @ O_i + P_ij @ V_j
                
                # 更新统计量
                m_i = m_ij
                l_i = l_ij
            
            # 步骤 7: 在芯片上最终归一化
            O_i = torch.diag(1.0 / l_i) @ O_i
            
            # 步骤 8: 计算 logsumexp
            L_i = m_i + torch.log(l_i)
            
            # 步骤 9: 将 O_i 写回 HBM
            O[start_i:end_i, :] = O_i
            L[start_i:end_i] = L_i
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.Br = Br
        ctx.Bc = Bc
        
        return O, L
    
    @staticmethod
    def backward(ctx, dO, dL):
        raise NotImplementedError("Backward pass is not implemented")