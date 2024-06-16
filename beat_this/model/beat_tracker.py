from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from torch.distributions.exponential import Exponential
from rotary_embedding_torch import RotaryEmbedding
from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype
from einops import rearrange

import beat_this.model.roformer as roformer


class BeatThis(nn.Module):
    def __init__(
        self,
        spect_dim=128,
        total_dim=512,
        ff_mult=4,
        n_layers=6,
        head_dim=32,
        stem_dim=32,
        dropout={"input": 0.2, "frontend": 0.1, "transformer": 0.2},
    ):
        super().__init__()

        assert total_dim % head_dim == 0, "total_dim must be divisible by head_dim"
        n_heads = total_dim // head_dim
        rotary_embed = RotaryEmbedding(head_dim)

        self.input_dropout = nn.Dropout(dropout["input"])

        # create the frontend
        stem = nn.Sequential(
            OrderedDict(
                rearrange_tf=Rearrange("b t f -> b f t"),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(in_channels=1, out_channels=stem_dim,
                        kernel_size=(4, 3), stride=(4, 1), padding=(0, 1), padding_mode="zeros"),
                bn2d=nn.BatchNorm2d(stem_dim),
                activation=nn.GELU(),
            )
        )
        frontend_blocks = []
        _dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                nn.Sequential(
                    OrderedDict(
                        fpartial=PartialRoformer(  # frequency directed partial transformer
                            dim=_dim,
                            dim_head=head_dim,
                            n_head=_dim // head_dim,
                            direction="F",
                            rotary_embed=rotary_embed,
                            dropout=dropout["frontend"],
                        ),
                        tpartial=PartialRoformer(  # time directed partial transformer
                            dim=_dim,
                            dim_head=head_dim,
                            n_head=_dim // head_dim,
                            direction="T",
                            rotary_embed=rotary_embed,
                            dropout=dropout["frontend"],
                        ),
                        # conv block
                        conv2d=nn.Conv2d(in_channels=_dim, out_channels=_dim * 2,
                                kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)),
                        # out_channels : 64, 128, 256
                        # freqs : 16, 8, 4 (due to the stride=2)
                        norm=nn.BatchNorm2d(_dim*2),
                        activation=nn.GELU(),
                    )
                )
            )
            _dim = _dim * 2
        frontend_blocks = nn.Sequential(*frontend_blocks)
        concat = Rearrange("b c f t -> b t (c f)")
        last_linear = nn.Linear(_dim*4, total_dim)
        self.frontend = nn.Sequential(
            stem, frontend_blocks, concat, last_linear)

        # create the transformer blocks
        self.transformer_blocks = roformer.Transformer(dim=total_dim, depth=n_layers, heads=n_heads, attn_dropout=dropout["transformer"],
                                                       ff_dropout=dropout["transformer"], rotary_embed=rotary_embed, ff_mult=ff_mult, dim_head=head_dim, norm_output=True)

        # create the output heads
        self.task_heads = SumHead(total_dim)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.input_dropout(x)
        x = self.transformer_blocks(x)
        x = self.task_heads(x)
        return x


class PartialRoformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block either only across frequencies or only across time.
    """

    def __init__(self, dim, dim_head, n_head, direction, rotary_embed, dropout):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        self.direction = direction[0].lower()
        if not self.direction in 'ft':
            raise ValueError(f"direction must be F or T, got {direction}")
        self.attn = roformer.Attention(dim, heads=n_head, dim_head=dim_head,
                                       dropout=dropout, rotary_embed=rotary_embed)
        self.ff = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        if self.direction == 'f':
            pattern = "(b t) f c"
        elif self.direction == 't':
            pattern = "(b f) t c"
        x = rearrange(x, f"b c f t -> {pattern}")
        x = x + self.attn(x)
        x = x + self.ff(x)
        x = rearrange(x, f"{pattern} -> b c f t", b=b)
        return x


class SumHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x):
        beat_downbeat = self.beat_downbeat_lin(x)
        # separate beat from downbeat
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)
        # aggregate beats and downbeats prediction
        # autocast is necessary to avoid numerical issues causing NaNs
        with torch.autocast(beat.device.type, enabled=False):
            beat = beat.float() + downbeat.float()
        return {"beat":beat, "downbeat": downbeat}


if __name__ == "__main__":
    model = BeatThis()
    x = torch.randn(4, 1500, 128)
    y = model(x)
    print(y[0].shape, y[1].shape)
