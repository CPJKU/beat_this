### Adapted from https://github.com/lucidrains/BS-RoFormer/tree/main ######

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from functools import wraps
from jbt.model.activation_fn import get_activation_fn

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# norm

class RMSNorm(Module):
    def __init__(self, size, dim=-1):
        super().__init__()
        self.scale = size ** 0.5
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim) * self.scale * self.gamma

# attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        activation = "gelu",
        dim_out = None,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        dim_inner = int(dim * mult)
        self.activation = get_activation_fn(activation, dim)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConvForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        activation = "gelu",
        size = 3,
        arch = 'two_conv',
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.activation = get_activation_fn(activation, dim)
        if arch == 'two_conv':
            self.net = nn.Sequential(
                RMSNorm(dim),
                nn.Conv1d(dim, dim_inner, size, padding='same'),
                self.activation,
                nn.Dropout(dropout),
                nn.Conv1d(dim_inner, dim, size, padding='same'),
                nn.Dropout(dropout)
            )
        elif arch == 'point_depth_point':
            self.net = nn.Sequential(
                RMSNorm(dim),
                nn.Conv1d(dim, dim_inner, 1),
                self.activation,
                nn.Conv1d(dim_inner, dim_inner, size, padding='same', groups=dim_inner),
                self.activation,
                nn.Dropout(dropout),
                nn.Conv1d(dim_inner, dim, 1),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError("unknown conv block architecture %r" % arch)

    def forward(self, x):
        x = self.net[0](x)
        x = x.transpose(-1, -2)
        x = self.net[1:](x)
        x = x.transpose(-1, -2)
        return x

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        gating = True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(dropout = dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        if gating:
            self.to_gates = nn.Linear(dim, heads)
        else:
            self.to_gates = None

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        if exists(self.to_gates):
            gates = self.to_gates(x)
            out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        activation = "gelu",
        gating = True,
        return_all_layers = False,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, activation = activation)
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, gating = gating),
                ff,
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.return_all_layers = return_all_layers

    def forward(self, x):
        if self.return_all_layers:
            attn_outs = []
        for attn, ff in self.layers:
            attn_x = attn(x)
            if self.return_all_layers:
                attn_outs.append(attn_x)
            x = attn_x + x
            x = ff(x) + x
        x = self.norm(x)
        if self.return_all_layers:
            return x, attn_outs
        else:
            return x


def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale

    def forward(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.
        )
