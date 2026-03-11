import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================================================================
#  RoPE Utilities
# ===========================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the cosine and sine frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)  # (seq_len, dim / 2)
    freqs_sin = torch.sin(freqs)  # (seq_len, dim / 2)
    return freqs_cos, freqs_sin


def apply_rotary_emb(q, k, freqs_cos, freqs_sin):
    """Applies Rotary Position Embedding to queries and keys."""
    # q, k shapes: (B, num_heads, N, head_dim)
    # freqs_cos, freqs_sin shapes: (N, head_dim // 2)

    # Reshape q and k to split the head dimension into pairs
    q_ = q.reshape(*q.shape[:-1], -1, 2)  # (B, num_heads, N, head_dim // 2, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 2)

    q_0, q_1 = q_[..., 0], q_[..., 1]
    k_0, k_1 = k_[..., 0], k_[..., 1]

    # Expand freqs to match q and k shapes
    fc = freqs_cos.view(1, 1, q.shape[2], -1)
    fs = freqs_sin.view(1, 1, q.shape[2], -1)

    # Apply the rotation matrix
    q_out_0 = q_0 * fc - q_1 * fs
    q_out_1 = q_0 * fs + q_1 * fc
    k_out_0 = k_0 * fc - k_1 * fs
    k_out_1 = k_0 * fs + k_1 * fc

    # Re-stack and flatten back to original shape
    q_out = torch.stack([q_out_0, q_out_1], dim=-1).flatten(3)
    k_out = torch.stack([k_out_0, k_out_1], dim=-1).flatten(3)

    return q_out, k_out


# ===========================================================================
#  Model Baseline: Transformer with RoPE
# ===========================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"Attention dimension ({dim}) must be divisible by num_heads ({num_heads})."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cos, freqs_sin):
        B, N, C = x.shape

        # Extract Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (B, num_heads, N, head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Standard Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Mix values and project
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cos, freqs_sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointTransformerClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 in_channels=3,
                 dim=216,
                 depth=4,
                 heads=6,
                 mlp_ratio=2.0,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 max_seq_len=2048):  # Added max_seq_len for RoPE initialization
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, dim)
        )

        # Precompute RoPE frequencies
        head_dim = dim // heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=drop_rate,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # NOTE: Expects x to be pre-ordered and shape (B, N, 3)
        if x.shape[1] == 3 and x.shape[2] > 3:
            x = x.transpose(1, 2).contiguous()

        N = x.shape[1]

        # Slice RoPE frequencies up to current sequence length
        freqs_cos = self.freqs_cos[:N]
        freqs_sin = self.freqs_sin[:N]

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)

        x = self.norm(x)
        x = x.max(dim=1)[0]
        return self.head(x)


# ===========================================================================
#  Model 1: The Global MLP Baseline (Modularized)
# ===========================================================================

class FourierFeatureMap(nn.Module):
    def __init__(self, in_features=3, num_bands=4, scale=10.0):
        super().__init__()
        self.register_buffer('B_matrix', torch.randn(in_features, num_bands * in_features) * scale)

    def forward(self, x):
        x_proj = (2. * math.pi * x) @ self.B_matrix
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.drop(self.act(self.linear1(self.norm(x))))


class GlobalMLPClassifier(nn.Module):
    def __init__(self,
                 num_classes=40,
                 num_points=1024,
                 in_channels=3,
                 num_bands=4,
                 mlp_dims=[512, 256, 128],
                 dropout_rates=[0.5, 0.5, 0.3]):
        super().__init__()
        assert len(mlp_dims) == len(dropout_rates), "mlp_dims and dropout_rates must match in length."

        self.num_points = num_points
        self.fourier_map = FourierFeatureMap(in_features=in_channels, num_bands=num_bands, scale=10.0)

        self.input_dim = num_points * (in_channels * num_bands * 2)

        blocks = []
        current_dim = self.input_dim
        for dim, drop in zip(mlp_dims, dropout_rates):
            blocks.append(MLPResidualBlock(current_dim, dim, drop=drop))
            current_dim = dim

        self.blocks = nn.Sequential(*blocks)
        self.final_norm = nn.LayerNorm(current_dim)
        self.head = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        # NOTE: Expects x to be pre-ordered and shape (B, N, 3)
        if x.shape[1] == 3 and x.shape[2] > 3:
            x = x.transpose(1, 2).contiguous()

        B, N, _ = x.shape
        assert N == self.num_points, f"GlobalMLP strictly requires N={self.num_points} points, got {N}."

        x = self.fourier_map(x)
        x = x.view(B, -1)

        x = self.blocks(x)
        x = self.final_norm(x)
        return self.head(x)