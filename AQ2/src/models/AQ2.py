"""
AlphaQubit 2 (AQ2) 神经网络解码器 —— 完整 PyTorch 实现
========================================================

论文: "A Scalable Real-Time Neural Decoder for Topological Quantum Codes"
      arXiv:2512.07737v2  (DeepMind / Google Quantum AI, 2026)

架构概述 (论文 A.1.1, Figure S2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本文件实现了 AQ2-full（高精度变体），层序列为:

    RNN → RNN → 3×Transformer → RNN → 3×Transformer → RNN → 3×Transformer → RNN

每层处理一个"时间 chunk"（K=6 个连续周期的压缩表示），RNN 将上一 chunk
的隐藏状态与新输入融合，Transformer 在空间维度上做稳定子之间的自注意力。

组件清单
~~~~~~~~
1. apply_rope()             — 旋转位置编码 (RoPE, A.1.2)
2. MultiHeadSelfAttentionKeySize — 多头自注意力 (可变 key_size)
3. StabilizerEmbedderLite   — 稳定子嵌入层 (A.1.2)
4. LightweightRNNCell       — 轻量 RNN 单元 (A.1.1, Figure S1b)
5. SpatialTransformerLayer  — 空间混合 Transformer 层 (A.1.1, Figure S1a)
6. AQ2Core                  — 核心计算引擎 (5 个 RNN + 3 组 Transformer)
7. CrossAttentionLayer      — 交叉注意力层 (用在 Readout 中)
8. ReadoutNetwork           — 读出网络 (A.1.1)
9. AuxiliaryHeads           — 四个辅助预测头 (A.2.3, A.3.3)  [新增]
10. GoogleDecoder            — 顶层模型 (GoogleDecoder = AQ2)

辅助损失说明 (A.2.3, A.3.3, Table S3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
训练时除了最终的 1 个 BCE 损失 (权重 1.2)，还计算 4 个辅助损失:

  ① pseudo_intermediate_result     (权重 1)  — 输入=伪终止稳定子嵌入
  ② noiseless                      (权重 1)  — 输入=regular state
  ③ noiseless_difference           (权重 1)  — 输入=regular state
  ④ noiseless_to_intermediate_diff (权重 8)  — 输入=伪终止稳定子嵌入

这 4 个头在模型中以 AuxiliaryHeads 实现，损失在 train_AQ2.py 中计算。

依赖
~~~~
- PyTorch >= 2.0
- src.data.google_data_utils.STABILIZER_LOCATIONS (码距 → 稳定子坐标映射)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.google_data_utils import STABILIZER_LOCATIONS


# =========================================================================
# 1. 旋转位置编码 (RoPE) — 论文 A.1.2
# =========================================================================

def apply_rope(q, k, pos_x, pos_y):
    """对 query 和 key 应用 2D 旋转位置编码 (RoPE)。

    论文 A.1.2: "For spatial self-attention we use Rotary Position Embedding
    (RoPE) of the physical qubit space coordinates. Half of the channels use
    the x coordinate, half use the y coordinate."

    实现细节
    ~~~~~~~~
    - K = 32 个 key 维度 → 16 对 (2 个元素形成一对旋转)
    - 前 8 对使用 x 坐标 (px), 后 8 对使用 y 坐标 (py)
    - 频率衰减: 1/10000^(2i/K), i=0..half-1
    - 旋转操作: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)

    Args:
        q: query 张量  (B, H, N, K)
        k: key 张量    (B, H, N, K)
        pos_x: x 坐标  (N,)  float32 — 归一化到 [-0.5, 0.5]
        pos_y: y 坐标  (N,)  float32 — 归一化到 [-0.5, 0.5]

    Returns:
        (q_rotated, k_rotated) — 旋转后的 query 和 key
    """
    # K 维度的一半: K=32 → half_k=32, half=16
    half_k = q.shape[-1] // 2       # query/key 的总维度 (应该是偶数)
    half = half_k // 2               # 旋转对的数量 (= 维度/4)

    # 频率: 1 / 10000^(2i / K), i=0..half-1  (论文标准 RoPE 频率)
    i = torch.arange(0, half, device=q.device, dtype=q.dtype)
    # freq = 1/10000^(2i / (2*half)) = 1/10000^(i/half).  The denominator is
    # 2*half (=K=32) not half (=K/2=16) so that the full span of 16 frequency
    # pairs covers the range [1/10000^0, 1/10000^1] — the standard grouped-RoPE
    # convention used by the paper (A.1.2).
    freq = 1.0 / (10000 ** (i / half))       # (half,)

    # 位置坐标扩展到正确的广播形状
    # pos_x, pos_y: (N,) → (1, 1, N, 1)  用于广播到 (B, H, N, half)
    px = pos_x.to(dtype=q.dtype, device=q.device).view(1, 1, -1, 1)
    py = pos_y.to(dtype=q.dtype, device=q.device).view(1, 1, -1, 1)
    freq = freq.view(1, 1, 1, -1)                                # (1, 1, 1, half)

    # 旋转角度: 前 8 对用 x 坐标 * 频率, 后 8 对用 y 坐标 * 频率
    # theta 形状: (1, 1, N, 16) — 每个稳定子位置有 16 个旋转角
    theta = torch.cat([px * freq, py * freq], dim=-1)
    cos = theta.cos()   # 预计算余弦
    sin = theta.sin()   # 预计算正弦

    def rotate(x):
        """对单个张量 (q 或 k) 应用 RoPE 旋转。

        算法
        ~~~~
        1. 将最后维度 K 重组为 (half_k, 2) = (16, 2)
        2. 每对 (x0, x1) 旋转角度 θ:
           x0' = x0·cos(θ) - x1·sin(θ)
           x1' = x0·sin(θ) + x1·cos(θ)
        3. 恢复到原始形状
        """
        # (B, H, N, K) → (B, H, N, 16, 2)
        x = x.reshape(*x.shape[:-1], half_k, 2)
        x0, x1 = x[..., 0], x[..., 1]      # 取出旋转对的两个分量
        # 应用 2D 旋转
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        # 合并并恢复到 (B, H, N, K)
        return torch.stack([x0_new, x1_new], dim=-1).reshape(q.shape)

    return rotate(q), rotate(k)


# =========================================================================
# 2. 多头自注意力 (可变 key_size)
# =========================================================================

class MultiHeadSelfAttentionKeySize(nn.Module):
    """多头自注意力，支持任意 key_size (不一定 = d_model // nhead)。

    论文未明确命名此组件，但从 A.1.1 的 "多头自注意力" 描述 + attn_key_size=32
    参数可知，inner_dim = nhead × key_size (而非标准的 d_model)。

    Args:
        d_model:  嵌入维度 (如 512)
        nhead:    注意力头数 (如 16)
        key_size: 每头的 key 维度 (如 32) → inner = nhead × key_size = 512
        dropout:  attention dropout 概率
    """

    def __init__(self, d_model: int, nhead: int, key_size: int, dropout: float):
        super().__init__()
        # 参数校验: 所有维度必须为正
        if d_model <= 0 or nhead <= 0 or key_size <= 0:
            raise ValueError("d_model, nhead, and key_size must be positive")
        self.d_model = d_model
        self.nhead = nhead
        self.key_size = key_size
        # inner 维度: nhead × key_size (如 16×32=512)
        self.inner = nhead * key_size
        # QKV 联合投影: d_model → 3×inner (查询、键、值合并在一个 Linear 中)
        self.qkv = nn.Linear(d_model, 3 * self.inner, bias=True)
        # 输出投影: inner → d_model
        self.out = nn.Linear(self.inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """前向传播。

        流程
        ~~~~
        1. QKV 投影 → 拆分为 Q, K, V
        2. 对 Q, K 应用 RoPE 位置编码
        3. Scaled Dot-Product Attention
        4. 拼接多头输出并投影

        Args:
            x:     (B, N, d_model) — 稳定子表示
            pos_x: (N,) — RoPE x 坐标
            pos_y: (N,) — RoPE y 坐标

        Returns:
            (B, N, d_model) — 注意力输出
        """
        bsz, n, _ = x.shape

        # 1. QKV 联合投影: (B, N, d_model) → (B, N, 3×inner)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # 沿最后一维均分为 3 份

        # 2. 重塑为多头格式: (B, N, inner) → (B, N, nhead, key_size) → (B, nhead, N, key_size)
        q = q.view(bsz, n, self.nhead, self.key_size).transpose(1, 2)
        k = k.view(bsz, n, self.nhead, self.key_size).transpose(1, 2)
        v = v.view(bsz, n, self.nhead, self.key_size).transpose(1, 2)

        # 3. 应用 RoPE (仅 Q 和 K, V 不参与位置编码)
        q, k = apply_rope(q, k, pos_x, pos_y)

        # 4. Scaled Dot-Product Attention
        scale = float(self.key_size) ** -0.5   # 缩放因子: 1/√key_size
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)      # 沿 key 维度做 softmax
        attn = self.dropout(attn)                # Attention dropout

        # 5. 加权求和: (B, H, N, key_size)
        out = torch.matmul(attn, v)

        # 6. 拼接多头: (B, H, N, K) → (B, N, H, K) → (B, N, H×K)
        out = out.transpose(1, 2).contiguous().view(bsz, n, self.inner)

        # 7. 输出投影: (B, N, inner) → (B, N, d_model)
        out = self.out(out)
        return out


# =========================================================================
# 3. 稳定子嵌入层 — 论文 A.1.2
# =========================================================================

class StabilizerEmbedderLite(nn.Module):
    """稳定子嵌入层。

    论文 A.1.2 描述: "Stabilizers are embedded through a linear embedding.
    For surface codes we add the embeddings of the measurement and event
    for each stabilizer. [...] We also add a normalized position encoding
    linearly embedding the normalized x and y coordinates for each qubit
    relative to the current code block size."

    嵌入公式
    ~~~~~~~~
    S_n = Linear(m_n) + Linear(e_n) + Embed(i_n) + Linear(px) + Linear(py)
    然后过一个 2 层 ResNet 块。

    Args:
        num_stabilizers: 稳定子数量 (= d² - 1)
        d_model:         嵌入维度
        dropout:         dropout 概率
    """

    def __init__(self, num_stabilizers, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 测量投影: 标量 (B, N, 1) → (B, N, D)
        self.proj_measurement = nn.Linear(1, d_model)

        # 检测事件投影: 标量 (B, N, 1) → (B, N, D)
        self.proj_detection = nn.Linear(1, d_model)

        # 稳定子索引嵌入: (N,) → (N, D), 然后广播到 batch
        self.embed_index = nn.Embedding(num_stabilizers, d_model)

        # ResNet 块: 两个 Linear + GELU + 残差连接
        self.resnet_linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()              # 论文使用的激活函数
        self.resnet_linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)          # 论文 A.1.1: "RMSNorm layer"

        # 绝对位置编码: x 和 y 坐标各自独立嵌入
        # 论文 A.1.2: "the x coordinate is always -0.5 (0.5) on the left
        # (right) edge, regardless of code distance"
        self.proj_pos_x = nn.Linear(1, d_model)
        self.proj_pos_y = nn.Linear(1, d_model)

    def forward(self, measurements, detection_events, stabilizer_indices, pos_x, pos_y):
        """嵌入稳定子测量和检测事件。

        流程
        ~~~~
        1. 确保输入有最后维度 (2D → 3D)
        2. 投影测量值、检测事件
        3. 嵌入稳定子索引并广播到 batch
        4. 投影绝对位置坐标
        5. 求和所有嵌入 → x
        6. 过 ResNet 块: x = norm(x) → linear1 → GELU → dropout → linear2 → +残差

        Args:
            measurements:      (B, N) 或 (B, N, 1) — 累积异或测量值
            detection_events:  (B, N) 或 (B, N, 1) — 检测事件
            stabilizer_indices: (N,) — 稳定子索引 (0..N-1)
            pos_x: (N,) — 归一化 x 坐标
            pos_y: (N,) — 归一化 y 坐标

        Returns:
            (B, N, d_model) — 嵌入后的稳定子表示
        """
        # 1. 确保输入有最后维度 (标量 → 1 维向量)
        if measurements.dim() == 2:
            measurements = measurements.unsqueeze(-1)    # (B, N) → (B, N, 1)
        if detection_events.dim() == 2:
            detection_events = detection_events.unsqueeze(-1)

        # 2. 投影: 每个标量 → d_model 维向量
        m_emb = self.proj_measurement(measurements)      # (B, N, D)
        e_emb = self.proj_detection(detection_events)    # (B, N, D)

        # 3. 稳定子索引嵌入: (N, D) → (1, N, D) → (B, N, D)
        i_emb = self.embed_index(stabilizer_indices)     # (N, D)
        i_emb = i_emb.unsqueeze(0).expand(measurements.size(0), -1, -1)

        # 4. 绝对位置编码: x 和 y 独立投影后相加
        #    pos_x, pos_y: (N,) → (N, 1) → Linear → (N, D) → (1, N, D)
        px_emb = self.proj_pos_x(pos_x.unsqueeze(-1))    # (N, D)
        py_emb = self.proj_pos_y(pos_y.unsqueeze(-1))    # (N, D)
        pos_emb = px_emb.unsqueeze(0) + py_emb.unsqueeze(0)  # (1, N, D) 广播

        # 5. 求和: m + e + i + pos — 论文 A.1.2 公式
        x = m_emb + e_emb + i_emb + pos_emb              # (B, N, D)

        # 6. ResNet 块: Pre-LN 风格 (norm → sublayers → +residual)
        residual = x                                      # 保存残差
        x = self.norm(x)                                  # RMSNorm 归一化
        x = self.resnet_linear1(x)                        # 第一层 Linear
        x = self.activation(x)                            # GELU 激活
        x = self.dropout(x)                               # Dropout
        x = self.resnet_linear2(x)                        # 第二层 Linear
        x = x + residual                                  # 残差连接

        return x


# =========================================================================
# 4. 轻量 RNN 单元 
# =========================================================================

class LightweightRNNCell(nn.Module):
    """轻量 RNN 单元。




    """

    def __init__(self, d_model: int):
        super().__init__()
        # 拼接 [state, input] → d_model: 输入维度 2×d_model
        self.project = nn.Linear(d_model * 2, d_model)
        self.norm = nn.RMSNorm(d_model)          # 输出归一化
        # 论文: "initializing the weights of the projection state to a
        # random orthogonal matrix, we observed better training"
        nn.init.orthogonal_(self.project.weight)

    def forward(self, state, new_input=None):
        """前向传播。

        如果 new_input 为 None, 直接返回 state (恒等映射，用于推理时
        没有新输入的纯状态传播)。

        Args:
            state:     (B, N, D) — 上一时间步的状态
            new_input: (B, N, D) — 当前时间步的新输入, 或 None

        Returns:
            (B, N, D) — 更新后的状态
        """
        if new_input is None:
            # 无新输入时恒等传播（推理场景，如流式解码的纯 forward pass）
            h = state
        else:
            # 1. 拼接状态和新输入: (B, N, 2D)
            h = torch.cat([state, new_input], dim=-1)
            # 2. 线性投影: (B, N, 2D) → (B, N, D)
            h = self.project(h)
            # 3. GELU 激活 + RMSNorm
            h = F.gelu(h)
            h = self.norm(h)
        return h


# =========================================================================
# 5. 空间混合 Transformer 层 — 论文 A.1.1, Figure S1a
# =========================================================================

class SpatialTransformerLayer(nn.Module):
    """空间混合 Transformer 层。

    论文 A.1.1: "Spatial mixing Transformer layers (Fig. S1a) consist of a
    RMSNorm normalization layer followed by multi-head self-attention among
    stabilizer embeddings with a parallel residual connection. After a
    further normalization, a gated dense block computes a further update on
    the residuated activations."

    结构 (Pre-LN)
    ~~~~~~~~~~~~~~
    1. RMSNorm → MultiHeadSelfAttention → Dropout → +残差
    2. RMSNorm → GLU (Linear×2 → chunk → GELU×gate) → Dropout → Linear → +残差

    GLU 宽度约定
    ~~~~~~~~~~~~
    linear1 输出 = 2 × dim_feedforward (value + gate 各 dim_feedforward)。
    论文 Table S1 "Widening 4" 指 linear1 输出 = 4×d_model，即 dim_feedforward
    = 2×d_model。d_model=512 时 dim_feedforward=1024，linear1 输出 2048。

    Args:
        d_model:        嵌入维度
        nhead:          注意力头数
        dim_feedforward: FFN 中间维度 (GLU 后 value/gate 各自的维度；linear1
                         输出 = 2×dim_feedforward)
        dropout:        dropout 概率
        key_size:       每头 key 维度
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, key_size):
        super().__init__()

        # --- 注意力子层 (Pre-LN) ---
        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = MultiHeadSelfAttentionKeySize(
            d_model=d_model, nhead=nhead,
            key_size=key_size, dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        # --- GLU 前馈子层 (Pre-LN) ---
        # GLU = Gated Linear Unit: 将输入线性投影到 2×dim_ff,
        #   前半为激活值 (过GELU), 后半为门控值 (sigmoid隐式), 逐元素相乘
        self.norm2 = nn.RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)  # 2× 用于 GLU
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)      # 投影回 d_model

    def forward(self, src, pos_x=None, pos_y=None):
        """前向传播。

        Args:
            src:   (B, N, d_model) — 输入表示
            pos_x: (N,) — RoPE x 坐标, 或 None
            pos_y: (N,) — RoPE y 坐标, 或 None

        Returns:
            (B, N, d_model) — 更新后的表示
        """
        # 子层 1: 多头自注意力 + 残差 (Pre-LN)
        src2 = self.norm1(src)                    # Pre-LN 归一化
        src2 = self.self_attn(src2, pos_x, pos_y) # 多头自注意力
        src = src + self.dropout1(src2)           # 残差连接

        # 子层 2: GLU 前馈网络 + 残差 (Pre-LN)
        src2 = self.norm2(src)                    # Pre-LN 归一化
        src2 = self.linear1(src2)                 # (B, N, 2×dim_ff)
        src2, gate = src2.chunk(2, dim=-1)        # 拆分为激活和门控
        src2 = F.gelu(src2) * gate               # GELU(激活) ⊙ 门控  (= GLU)
        src2 = self.dropout2(src2)                # Dropout
        src2 = self.linear2(src2)                 # 投影回 d_model
        src = src + src2                          # 残差连接

        return src


# =========================================================================
# 6. AQ2 核心计算引擎 — 论文 A.1.1, Figure S2
# =========================================================================

class AQ2Core(nn.Module):
    """AQ2 的核心计算引擎。

    层序列 (论文 Figure S2)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    RNN1 → RNN2 → 3×Transformer₁ → RNN3 → 3×Transformer₂ → RNN4 → 3×Transformer₃ → RNN5

    5 个 RNN 将信息沿时间轴向前传播（对每个稳定子独立进行），
    3 组 Transformer（每组 3 层）在空间维度让稳定子之间交换信息。

    Args:
        d_model:        嵌入维度
        nhead:          注意力头数
        dim_feedforward: FFN 中间维度
        dropout:        dropout 概率
        key_size:       每头 key 维度
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, key_size):
        super().__init__()

        # ---- 5 个轻量 RNN (Figure S1b) ----
        self.rnn1 = LightweightRNNCell(d_model)
        self.rnn2 = LightweightRNNCell(d_model)
        self.rnn3 = LightweightRNNCell(d_model)
        self.rnn4 = LightweightRNNCell(d_model)
        self.rnn5 = LightweightRNNCell(d_model)

        # ---- 3 组 Transformer, 每组 3 层 (Figure S1a) ----
        t_args = (d_model, nhead, dim_feedforward, dropout, key_size)
        self.transformer_block1 = nn.ModuleList([
            SpatialTransformerLayer(*t_args) for _ in range(3)
        ])
        self.transformer_block2 = nn.ModuleList([
            SpatialTransformerLayer(*t_args) for _ in range(3)
        ])
        self.transformer_block3 = nn.ModuleList([
            SpatialTransformerLayer(*t_args) for _ in range(3)
        ])

    def forward(self, previous_states, chunk_embedding, pos_x=None, pos_y=None):
        """处理一个时间 chunk。

        时序传播
        ~~~~~~~~
        - previous_states[0..4]: 上一个 chunk 结束后 5 个 RNN 的状态
        - chunk_embedding: 当前 chunk 的时间压缩嵌入 (B, N, D)
        - 返回值: (final_state, new_states[0..4]) 供下一个 chunk 使用

        流程
        ~~~~
        RNN1(prev0, chunk) → s1 → RNN2(prev1, s1) → s2
        → 3×Transformer₁(s2) → s2'
        → RNN3(prev2, s2') → s3
        → 3×Transformer₂(s3) → s3'
        → RNN4(prev3, s3') → s4
        → 3×Transformer₃(s4) → s4'
        → RNN5(prev4, s4') → s5  (最终输出)

        Args:
            previous_states:  list[5] of (B, N, D) — 上一个 chunk 的 RNN 状态
            chunk_embedding:  (B, N, D) — 当前 chunk 的压缩嵌入
            pos_x:            (N,) — RoPE x 坐标
            pos_y:            (N,) — RoPE y 坐标

        Returns:
            (state, new_states) where state=(B,N,D), new_states=list[5]of(B,N,D)
        """
        new_states = []  # 收集 5 个 RNN 的输出状态

        # 阶段 1: RNN1 → RNN2 (时间更新，无 Transformer)
        state = self.rnn1(previous_states[0], chunk_embedding)
        new_states.append(state)                                       # 保存 RNN1 输出

        state = self.rnn2(previous_states[1], state)
        new_states.append(state)                                       # 保存 RNN2 输出

        # 阶段 2: 3 层空间 Transformer (第 1 组)
        for layer in self.transformer_block1:
            state = layer(state, pos_x, pos_y)

        # 阶段 3: RNN3 (时间更新)
        state = self.rnn3(previous_states[2], state)
        new_states.append(state)                                       # 保存 RNN3 输出

        # 阶段 4: 3 层空间 Transformer (第 2 组)
        for layer in self.transformer_block2:
            state = layer(state, pos_x, pos_y)

        # 阶段 5: RNN4 (时间更新)
        state = self.rnn4(previous_states[3], state)
        new_states.append(state)                                       # 保存 RNN4 输出

        # 阶段 6: 3 层空间 Transformer (第 3 组)
        for layer in self.transformer_block3:
            state = layer(state, pos_x, pos_y)

        # 阶段 7: RNN5 (最终状态更新)
        state = self.rnn5(previous_states[4], state)
        new_states.append(state)                                       # 保存 RNN5 输出

        return state, new_states


# =========================================================================
# 7. 交叉注意力层 — 用在 ReadoutNetwork 中
# =========================================================================

class CrossAttentionLayer(nn.Module):
    """交叉注意力层。

    论文 A.1.1: "each logical observable's representation does
    cross-attention to the final per-stabilizer representation."
    "After two cross-attention Transformer layers..."

    这里的 CrossAttentionLayer 实现的是 Attention 子层 (不含 FFN)。
    论文后文说"两个残差密集层处理"——那两个 dense 层放在 ReadoutNetwork
    的末尾，不在这里。

    结构
    ~~~~
    1. RMSNorm(query) + RMSNorm(kv)
    2. Q = query 投影, K,V = kv 投影
    3. Scaled Dot-Product Cross-Attention
    4. 输出投影 + 残差 (query + out)

    Args:
        d_model:  嵌入维度
        nhead:    注意力头数
        key_size: 每头 key 维度
        dropout:  dropout 概率
    """

    def __init__(self, d_model, nhead, key_size, dropout):
        super().__init__()
        # 独立归一化 query 和 key/value
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)

        # query 投影: d_model → nhead × key_size
        self.q_proj = nn.Linear(d_model, nhead * key_size)
        # key 投影
        self.k_proj = nn.Linear(d_model, nhead * key_size)
        # value 投影
        self.v_proj = nn.Linear(d_model, nhead * key_size)
        # 输出投影: nhead × key_size → d_model
        self.out = nn.Linear(nhead * key_size, d_model)

        self.nhead = nhead
        self.key_size = key_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv):
        """交叉注意力前向传播。

        query 是单个可观测量的表示 (B, 1, D), kv 是全体稳定子的最终表示 (B, N, D)。
        这种不对称的设计让每个逻辑可观测量可以有选择性地关注不同稳定子。

        Args:
            query: (B, 1, d_model) — 可观测量的表示
            kv:    (B, N, d_model) — 稳定子的最终表示

        Returns:
            (B, 1, d_model) — 更新后的可观测量表示
        """
        B, N, D = kv.shape

        # Pre-LN: 分别归一化 query 和 key/value
        q = self.norm_q(query)          # (B, 1, D)
        kv_normed = self.norm_kv(kv)    # (B, N, D)

        # 投影 + 多头拆分
        # Q: (B, 1, D) → (B, 1, H×K) → (B, H, 1, K)
        q = self.q_proj(q).view(B, 1, self.nhead, self.key_size).transpose(1, 2)
        # K,V: (B, N, D) → (B, N, H×K) → (B, H, N, K)
        k = self.k_proj(kv_normed).view(B, N, self.nhead, self.key_size).transpose(1, 2)
        v = self.v_proj(kv_normed).view(B, N, self.nhead, self.key_size).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.key_size ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, 1, N)
        attn = torch.softmax(attn, dim=-1)                     # 沿稳定子维度 softmax
        attn = self.dropout(attn)

        # 加权求和: (B, H, 1, K)
        out = torch.matmul(attn, v)

        # 拼接多头: (B, 1, H, K) → (B, 1, H×K)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.nhead * self.key_size)
        out = self.out(out)    # 输出投影: (B, 1, D)

        return query + out     # 残差连接


# =========================================================================
# 8. 读出网络 — 论文 A.1.1
# =========================================================================

class ReadoutNetwork(nn.Module):
    """读出网络。

    论文 A.1.1: "The readout network pools the final per-stabilizer
    representation by averaging. We replicate the representation for each
    logical observable to be predicted (only one at inference time) and add
    a learned embedding for each. Each logical observable's representation
    does cross-attention to the final per-stabilizer representation.
    After two cross-attention Transformer layers, each logical observable
    representation is processed by two residual dense layers, then
    projected to a single channel and passed through a logistic activation."

    流程
    ~~~~
    1. mean-pool 稳定子表示 → (B, D)
    2. + 可观测量嵌入 → (B, 1, D)
    3. 交叉注意力 ×2 (到稳定子表示)
    4. 残差密集层 ×2
    5. 线性输出 → (B, 1)

    Args:
        d_model:  嵌入维度
        nhead:    注意力头数
        key_size: 每头 key 维度
        dropout:  dropout 概率
    """

    def __init__(self, d_model, nhead, key_size, dropout):
        super().__init__()
        # 可观测量嵌入: 2 种 (X 和 Z), 每种 d_model 维
        self.obs_embed = nn.Embedding(2, d_model)

        # 两个交叉注意力层
        self.cross_attn1 = CrossAttentionLayer(d_model, nhead, key_size, dropout)
        self.cross_attn2 = CrossAttentionLayer(d_model, nhead, key_size, dropout)

        # 两个残差密集层 (论文: "two residual dense layers")
        self.dense1 = nn.Linear(d_model, d_model)
        self.dense2 = nn.Linear(d_model, d_model)

        # 最终输出: 单通道 logit (sigmoid 在损失函数 BCEWithLogitsLoss 中)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, stabilizer_states, obs_idx=None):
        """前向传播。

        Args:
            stabilizer_states: (B, N, D) — 最终的稳定子表示
            obs_idx:           (B,) 或 None — 可观测量索引 (0=Z, 1=X), None 则默认 0

        Returns:
            (B, 1) — logit (未过 sigmoid, 由 BCEWithLogitsLoss 处理)
        """
        B, N, D = stabilizer_states.shape

        # 1. 平均池化: (B, N, D) → (B, D)
        pooled = stabilizer_states.mean(dim=1)

        # 2. 默认可观测量索引为 0 (Z 基)
        if obs_idx is None:
            obs_idx = torch.zeros(B, dtype=torch.long, device=pooled.device)

        # 3. 池化表示 + 可观测量嵌入: (B, D) → (B, 1, D)
        obs_rep = pooled.unsqueeze(1) + self.obs_embed(obs_idx).unsqueeze(1)

        # 4. 交叉注意力 ×2: 可观测量 (B,1,D) attend 到全体稳定子 (B,N,D)
        obs_rep = self.cross_attn1(obs_rep, stabilizer_states)
        obs_rep = self.cross_attn2(obs_rep, stabilizer_states)

        # 5. 残差密集层 ×2
        x = obs_rep.squeeze(1)                    # (B, D)
        x = x + F.gelu(self.dense1(x))            # 残差 1: x + GELU(Linear(x))
        x = x + F.gelu(self.dense2(x))            # 残差 2: x + GELU(Linear(x))

        # 6. 输出投影: (B, D) → (B, 1)
        return self.output_head(x)


# =========================================================================
# 9. 辅助预测头 — 论文 A.2.3, A.3.3  [2026-07-31 新增]
# =========================================================================

# 四个辅助 BCE 头的权重 (Table S3)
#   pseudo_intermediate_result      = 1
#   noiseless                       = 1
#   noiseless_difference            = 1
#   noiseless_to_intermediate_diff  = 8
#   final (主损失)                  = 1.2  (在 train_AQ2.py 中手动加权)
#
# 注意: 列表顺序为 AuxiliaryHeads.forward() 的返回顺序:
#   AUX_WEIGHTS[0] → 头①, [1] → 头②, [2] → 头③, [3] → 头④
AUX_WEIGHTS = (1.0, 1.0, 1.0, 8.0)


class AuxiliaryHeads(nn.Module):
    """四个独立的辅助预测头，按论文 A.3.3 路由输入。

    输入路由
    ~~~~~~~~
    论文 A.3.3 明确了两类不同的输入:
      - 头 ① (pseudo_intermediate)        → 伪终止稳定子测量的嵌入
      - 头 ④ (noiseless_to_inter_diff)    → 伪终止稳定子测量的嵌入
      - 头 ② (noiseless)                  → regular state (AQ2Core 输出)
      - 头 ③ (noiseless_diff)             → regular state (AQ2Core 输出)

    每个头的结构
    ~~~~~~~~~~~~
    RMSNorm(d_model) → Linear(d_model, d_model) → GELU → Linear(d_model, 1)
    → squeeze → (B,) 标量 logit

    预测目标
    ~~~~~~~~
    ① pseudo_obs[t]:             伪终止 observable (非恒0, 误差累积)
    ② noiseless[t]:              无噪声 observable (恒为 0, |0⟩ 初态)
    ③ XOR(noiseless[t], noiseless[t-1]): 无噪声差分 (恒为 0)
    ④ XOR(noiseless[t-1], pseudo[t]):     = pseudo[t] (因为 noiseless 恒为 0)

    Args:
        d_model: 嵌入维度 (与主模型一致)
    """

    def __init__(self, d_model: int):
        super().__init__()
        # ---- 头 ①, ④ — 输入 = 伪终止稳定子嵌入 ----
        self.pseudo_intermediate = self._make_head(d_model)         # 权重 1
        self.noiseless_to_inter_diff = self._make_head(d_model)     # 权重 8

        # ---- 头 ②, ③ — 输入 = regular state ----
        self.noiseless = self._make_head(d_model)                   # 权重 1
        self.noiseless_diff = self._make_head(d_model)              # 权重 1

    @staticmethod
    def _make_head(d_model: int) -> nn.Sequential:
        """构建单个辅助预测头的网络结构。

        两层 MLP + RMSNorm:
          RMSNorm → Linear(d, d) → GELU → Linear(d, 1)
        输出标量 logit (未过 sigmoid, 由 BCEWithLogitsLoss 处理)。
        """
        return nn.Sequential(
            nn.RMSNorm(d_model),           # 输入归一化
            nn.Linear(d_model, d_model),   # 隐藏层 (保持维度)
            nn.GELU(),                     # GELU 激活
            nn.Linear(d_model, 1),         # 输出投影 → 标量
        )

    def forward(self, main_state: torch.Tensor,
                aux_state: torch.Tensor = None):
        """对单个时间 chunk 应用全部四个辅助头。

        Heads ①④ (pseudo_intermediate / noiseless_to_inter_diff) receive
        *aux_state* — the AQ2 core output after a dedicated pass that feeds the
        fake-ending chunk.  Heads ②③ (noiseless / noiseless_diff) continue to
        use *main_state* — the regular AQ2 core output after the true chunk.

        Without *aux_state* (e.g. during evaluation) heads ①④ return zeros.

        Args:
            main_state: (B, N, D) — AQ2Core output after processing the real chunk
            aux_state:  (B, N, D) or None — AQ2Core output after fake-ending chunk

        Returns:
            Tuple[4 × (B,)] — 四个标量 logit, 顺序为: ①②③④
        """
        s_main = main_state.mean(dim=1)       # (B, D) — regular state pool
        if aux_state is not None:
            s_aux = aux_state.mean(dim=1)     # (B, D) — pseudo-state pool
            pred_1 = self.pseudo_intermediate(s_aux).squeeze(-1)
            pred_4 = self.noiseless_to_inter_diff(s_aux).squeeze(-1)
        else:
            s_aux = None  # unreachable when return_aux=True; safety
            pred_1 = torch.zeros(s_main.shape[0], device=s_main.device)
            pred_4 = torch.zeros(s_main.shape[0], device=s_main.device)

        pred_2 = self.noiseless(s_main).squeeze(-1)
        pred_3 = self.noiseless_diff(s_main).squeeze(-1)

        return (pred_1, pred_2, pred_3, pred_4)


# =========================================================================
# 10. GoogleDecoder (顶层模型) — 论文 Figure S2 的完整实现
# =========================================================================

class GoogleDecoder(nn.Module):
    """AlphaQubit 2 (AQ2) 完整解码器。

    这是论文的 GoogleDecoder 类，包含:
      - StabilizerEmbedderLite (嵌入层)
      - 时间压缩 (Linear: K×D → D)
      - AQ2Core (RNN+Transformer 计算引擎)
      - AuxiliaryHeads (4 个辅助头, 仅训练时激活)
      - ReadoutNetwork (最终预测)

    时间压缩
    ~~~~~~~~
    论文 Section 3.2: "we combine consecutive (typically 3 to 6) measurement
    cycles in a group, using a learned temporal compression, without loss of
    accuracy."

    对每个 K 轮组成的 chunk:
      1. 将 K 个 embedding 拼接: (B, N, K×D)
      2. 通过 Linear 投影回: (B, N, D)
      3. 送入 AQ2Core 处理

    Args:
        distances:      训练涉及的码距列表, 如 [3, 5, 7, 9, 11]
        num_rounds:     最大轮数 (用于确定 temporal 维度)
        d_model:        嵌入维度 (默认 512)
        nhead:          注意力头数 (默认 16)
        dim_feedforward: FFN 中间维度 (默认 1024 = 2×d_model, GLU 后 value+gate 各 1024)
        dropout:        dropout 概率 (默认 0.1)
        attn_key_size:  每头 key 维度 (默认 32)
        temporal_K:     时间压缩的 chunk 大小 (默认 6, 论文对表面码用 6)
    """

    def __init__(self,
                 distances: list,
                 num_rounds: int,
                 d_model: int = 512,
                 nhead: int = 16,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 attn_key_size: int = 32,
                 temporal_K: int = 6):
        super().__init__()
        self.distances = distances
        self.num_rounds = num_rounds
        self.K = temporal_K                        # 时间压缩因子

        # ---- 嵌入层 ----
        # 计算最大码距所需的稳定子数量 (d²-1)
        max_distance = max(distances)
        max_grid_dim = 2 * max_distance + 1
        max_stabilizers = len([
            y * max_grid_dim + x
            for (y, x) in STABILIZER_LOCATIONS[max_distance]
        ])
        self.embedder = StabilizerEmbedderLite(max_stabilizers, d_model, dropout)

        # ---- RoPE 坐标缓存 ----
        # 为每个参与训练的码距预先计算并缓存归一化的稳定子坐标
        # register_buffer 确保坐标随模型移动到正确的 device
        for d in distances:
            grid_dim = 2 * d + 1                                          # 网格大小
            # x 坐标: 归一化到 [-0.5, 0.5] — 论文 A.1.2
            px = torch.tensor(
                [x / grid_dim - 0.5 for (y, x) in STABILIZER_LOCATIONS[d]],
                dtype=torch.float32,
            )
            py = torch.tensor(
                [y / grid_dim - 0.5 for (y, x) in STABILIZER_LOCATIONS[d]],
                dtype=torch.float32,
            )
            self.register_buffer(f"pos_x_d{d}", px, persistent=False)
            self.register_buffer(f"pos_y_d{d}", py, persistent=False)

        # ---- 时间压缩 ----
        # K 个连续的 embedding 拼接后投影回 D 维
        self.temporal_compress = nn.Linear(d_model * self.K, d_model)

        # ---- 核心计算引擎 ----
        self.aq2_core = AQ2Core(
            d_model, nhead, dim_feedforward, dropout,
            key_size=attn_key_size,
        )

        # ---- 读出网络 ----
        self.readout = ReadoutNetwork(
            d_model, nhead,
            key_size=attn_key_size, dropout=dropout,
        )

        # ---- 辅助预测头 [2026-07-31 新增] ----
        self.aux_heads = AuxiliaryHeads(d_model)

    def forward(self, measurements, detection_events, stabilizer_indices,
                current_distance: int, obs_idx=None, mask=None,
                return_aux: bool = False,
                pseudo_measurements=None,        # (B, num_chunks, N) — 伪终止稳定子测量
                pseudo_detection_events=None,    # (B, num_chunks, N) — 伪终止检测事件
                aux_chunk_stride: int = 1):      # every N-th boundary gets an aux pass
        """前向传播。

        时序处理流程
        ~~~~~~~~~~~~
        for chunk_i in 0..ceil(T/K):
          1. 取 all_embs[:, i*K:(i+1)*K] → (B, K, N, D)
          2. 时间压缩: (B, N, K×D) → Linear → (B, N, D)
          3. AQ2Core(prev_states, compressed) → state
          4. 如果 return_aux: aux_heads(state, pseudo_embs[:, i])
        final: ReadoutNetwork(state) → logits

        Args:
            measurements:              (B, T, N) — 累积异或测量值
            detection_events:          (B, T, N) — 检测事件
            stabilizer_indices:        (N,)      — 稳定子索引
            current_distance:          int       — 当前码距 (决定RoPE坐标)
            obs_idx:                   (B,) 或 None — 可观测量索引
            mask:                      (B, T, N) 或 None — 稳定子遮盖 (A.2.2)
            return_aux:                bool — 是否返回辅助预测 (训练时 True)
            pseudo_measurements:       (B, C, N) 或 None — 伪终止测量
            pseudo_detection_events:   (B, C, N) 或 None — 伪终止检测事件

        Returns:
            if not return_aux:  logits (B, 1)
            if return_aux:      (logits, aux_tuple) where aux_tuple = 4 × (B, C)
        """
        B, T, N = measurements.shape           # batch, rounds, stabilizers
        D = self.embedder.d_model              # 嵌入维度
        C = (T + self.K - 1) // self.K         # number of temporal chunks

        # ---- 1. 加载当前码距的 RoPE 坐标 ----
        pos_x = getattr(self, f"pos_x_d{current_distance}")
        pos_y = getattr(self, f"pos_y_d{current_distance}")

        # ---- 2. 批量嵌入所有 T 轮 ----
        # 将所有 (B, T, N) 压平为 (B×T, N) 一次性嵌入, 再恢复为 (B, T, N, D)
        # 这样避免了逐轮循环, 大幅加速
        m_flat = measurements.reshape(B * T, N)            # (B×T, N)
        e_flat = detection_events.reshape(B * T, N)        # (B×T, N)
        all_embs = self.embedder(m_flat, e_flat, stabilizer_indices, pos_x, pos_y)
        all_embs = all_embs.reshape(B, T, N, D)            # (B, T, N, D)

        # ---- 3. 输入遮盖 (A.2.2) ----
        # 将 50% 的稳定子嵌入设为 0, 仅对 80% 的样本执行
        # 强迫模型学会从部分稳定子推断全局, 提高鲁棒性
        if mask is not None:
            all_embs = all_embs * mask.to(dtype=all_embs.dtype).unsqueeze(-1)

        # ---- 4. 预嵌入伪终止测量 (如果使用辅助头) ----
        # pseudo_measurements/detection_events: (B, C, N)
        # → 嵌入每个 chunk 边界 → (B, C, N, D)
        if return_aux:
            assert pseudo_measurements is not None, \
                "pseudo_measurements required when return_aux=True"
            assert pseudo_detection_events is not None, \
                "pseudo_detection_events required when return_aux=True"
            C = pseudo_measurements.shape[1]               # chunk 数量
            pm_flat = pseudo_measurements.reshape(B * C, N)   # (B×C, N)
            pe_flat = pseudo_detection_events.reshape(B * C, N)
            # 使用与主嵌入相同的 embedder (含 RoPE 坐标 + 位置编码)
            pseudo_embs = self.embedder(pm_flat, pe_flat, stabilizer_indices, pos_x, pos_y)
            pseudo_embs = pseudo_embs.reshape(B, C, N, D)     # (B, C, N, D)

        # ---- 5. 初始化 RNN 状态 (全零) ----
        # 论文: "Each state is initialised with a zero vector at the start of
        # the experiment"
        states_list = [
            torch.zeros(B, N, D, device=measurements.device)   # 5 个 RNN 的初始状态
            for _ in range(5)
        ]

        # ---- 6. 逐 chunk 时间循环 ----
        aux_preds: list = [] if return_aux else None
        chunk_idx = 0

        for i in range(0, T, self.K):                          # 步长 = temporal_K
            # 6a. 提取当前 chunk: (B, K, N, D), 最后一 chunk 可能不足 K
            chunk = all_embs[:, i:i+self.K, :, :]              # (B, actual_K, N, D)
            actual_K = chunk.shape[1]
            if actual_K < self.K:
                # 不足 K 时用零补齐 (处理截断 experiment)
                pad = torch.zeros(
                    B, self.K - actual_K, N, D,
                    device=chunk.device, dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, pad], dim=1)          # (B, K, N, D)

            # 6b. 时间压缩: (B, N, K, D) → (B, N, K×D) → Linear → (B, N, D)
            chunk = chunk.permute(0, 2, 1, 3)                  # (B, N, K, D)
            chunk = chunk.reshape(B, N, self.K * D)             # (B, N, K×D)
            chunk = self.temporal_compress(chunk)              # (B, N, D)

            # 6c. 送入 AQ2Core
            if return_aux:
                # Disable gradient checkpointing — aux heads need intermediate state
                state, states_list = self.aq2_core(
                    states_list, chunk, pos_x, pos_y)

                # --- Auxiliary heads ---
                # Heads ②③ (noiseless / noiseless_diff) use main state directly.
                # Heads ①④ (pseudo_intermediate / noiseless_to_inter_diff) use
                # an *auxiliary state* obtained by feeding the fake-ending cycle
                # (from pseudo_m/pseudo_e) as a single-round "chunk" through the
                # AQ2 core with the current RNN states — exactly the "terminate at
                # this cycle" branch per paper A.3.3.
                aux_state: torch.Tensor | None = None
                if (chunk_idx == 0
                        or chunk_idx == C - 1
                        or chunk_idx % aux_chunk_stride == 0):
                    # Build single-cycle fake chunk from pre-embedded pseudo data
                    fake_cycle = pseudo_embs[:, chunk_idx, :, :]  # (B, N, D)
                    fake_pad = torch.zeros(
                        B, self.K - 1, N, D,
                        device=fake_cycle.device, dtype=fake_cycle.dtype)
                    fake_input = torch.cat(
                        [fake_cycle.unsqueeze(1), fake_pad], dim=1)  # (B, K, N, D)
                    fake_input = fake_input.permute(0, 2, 1, 3)      # (B, N, K, D)
                    fake_input = fake_input.reshape(B, N, self.K * D)
                    fake_chunk_embed = self.temporal_compress(fake_input)  # (B, N, D)
                    # Core pass: true history through chunk c + fake final chunk
                    aux_state, _ = self.aq2_core(
                        states_list, fake_chunk_embed, pos_x, pos_y)
                aux_preds.append(self.aux_heads(state, aux_state))
                chunk_idx += 1
            else:
                # 评估/推理时使用 checkpointing 节省显存 (论文默认)
                state, states_list = torch.utils.checkpoint.checkpoint(
                    self.aq2_core, states_list, chunk, pos_x, pos_y,
                    use_reentrant=False,
                )

        # ---- 7. 最终预测 ----
        logits = self.readout(state, obs_idx=obs_idx)           # (B, 1)

        # ---- 8. 返回 ----
        if return_aux:
            # aux_preds: list[C] of tuple[4], 每个元素是 (B,)
            # → 转置为 tuple[4] of (B, C)
            aux = tuple(torch.stack(preds, dim=1) for preds in zip(*aux_preds))
            return logits, aux

        return logits
