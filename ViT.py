import math
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

@torch.no_grad()
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32, device=None):
    """
    2D sin-cos positional embeddings (ViT-compatible).
    Returns a (h*w, dim) tensor. Caller pads a zero row for the CLS token.
    """
    assert dim % 4 == 0, "dim must be multiple of 4 for sincos embeddings"
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    omega = torch.arange(dim // 4, device=device) / max(1, (dim // 4 - 1))
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).to(dtype=dtype)
    return pe  # (h*w, dim)

# core blocks
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        x = self.norm(x)                         # (B, N, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # 3 * (B, N, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, h, N, N)
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)             # (B, h, N, d)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)                  # (B, N, D)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 attn_drop=0.0, proj_drop=0.1, mlp_drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head,
                          attn_drop=attn_drop, proj_drop=proj_drop),
                FeedForward(dim, mlp_dim, drop=mlp_drop),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ViT
class ThaVit(nn.Module):
    """
    From-scratch ViT:
      - patchify (non-overlapping)
      - linear proj to dim
      - prepend CLS
      - add 2D sincos pos-emb
      - L x (LN -> MSA -> resid; LN -> MLP -> resid)
      - head on CLS
    """
    def __init__(self, *, image_size, patch_size, num_classes,
                 dim, depth, heads, mlp_dim, channels=3, dim_head=64,
                 attn_drop=0.0, proj_drop=0.1, mlp_drop=0.1):
        super().__init__()
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, "Image dims must be divisible by patch size"

        self.grid_h = ih // ph
        self.grid_w = iw // pw
        num_patches = self.grid_h * self.grid_w
        patch_dim = channels * ph * pw   # FIX: was hardcoded to 3 * ph * pw

        # patchify + linear projection
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=ph, pw=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # fixed sincos positions for the grid (buffer: not optimized, moves with module)
        pe = posemb_sincos_2d(self.grid_h, self.grid_w, dim)
        self.register_buffer("pos_embedding_grid", pe, persistent=False)  # (N, D)

        # encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, img):
        # img: (B, C, H, W)
        b = img.size(0)
        x = self.to_patch_embedding(img)                         # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)                   # (B, 1, D)

        # make (1, 1+N, D): pad a zero row for CLS then add to tokens+CLS
        pe_cls = torch.zeros(1, 1, x.size(-1), device=x.device, dtype=x.dtype)
        pe = torch.cat([pe_cls, self.pos_embedding_grid.unsqueeze(0).to(x)], dim=1)  # (1, 1+N, D)

        x = torch.cat([cls, x], dim=1) + pe
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]                                        # (B, D)
        return self.head(cls_out)                                # (B, num_classes)
