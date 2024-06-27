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
    """
    A neural network model for beat tracking. It is compose of three main components:
    - a frontend that processes the input spectrogram,
    - a series of transformer blocks that process the output of the frontend,
    - a head that produces the final beat and downbeat predictions.

    Args:
        spect_dim (int): The dimension of the input spectrogram (default: 128).
        transformer_dim (int): The dimension of the main transformer blocks (default: 512).
        ff_mult (int): The multiplier for the feed-forward dimension in the transformer blocks (default: 4).
        n_layers (int): The number of transformer blocks (default: 6).
        head_dim (int): The dimension of each attention head for the partial transformers in the frontend and the transformer blocks (default: 32).
        stem_dim (int): The out dimension of the stem convolutional layer (default: 32).
        dropout (dict): A dictionary specifying the dropout rates for different parts of the model
            (default: {"frontend": 0.1, "middle": 0.2, "transformer": 0.2}).
    """

    def __init__(
        self,
        spect_dim=128,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        head_dim=32,
        stem_dim=32,
        dropout={"frontend": 0.1,"middle": 0.2, "transformer": 0.2},
    ):
        super().__init__()

        assert transformer_dim % head_dim == 0, "transformer_dim must be divisible by head_dim"
        n_heads = transformer_dim // head_dim
        rotary_embed = RotaryEmbedding(head_dim)

        self.middle_dropout = nn.Dropout(dropout["middle"])

        # create the frontend
        stem = nn.Sequential(
            OrderedDict(
                rearrange_tf=Rearrange("b t f -> b f t"),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(in_channels=1, out_channels=stem_dim,
                        kernel_size=(4, 3), stride=(4, 1), padding=(0, 1), bias=False),
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
                        partial=PartialFTTrasformer(
                            dim=_dim,
                            dim_head=head_dim,
                            n_head=_dim // head_dim,
                            rotary_embed=rotary_embed,
                            dropout=dropout["frontend"],
                        ),
                        # conv block
                        conv2d=nn.Conv2d(in_channels=_dim, out_channels=_dim * 2,
                                kernel_size=(2, 3), stride=(2, 1), padding=(0, 1), bias=False),
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
        last_linear = nn.Linear(_dim*4, transformer_dim)
        self.frontend = nn.Sequential(
            stem, frontend_blocks, concat, last_linear)

        # create the transformer blocks
        self.transformer_blocks = roformer.Transformer(dim=transformer_dim, depth=n_layers, heads=n_heads, attn_dropout=dropout["transformer"],
                                                       ff_dropout=dropout["transformer"], rotary_embed=rotary_embed, ff_mult=ff_mult, dim_head=head_dim, norm_output=True)

        # create the output heads
        self.task_heads = SumHead(transformer_dim)

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
        x = self.middle_dropout(x)
        x = self.transformer_blocks(x)
        x = self.task_heads(x)
        return x
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # compiled models have _orig_mod in the state dict keys, remove it
        for key in list(state_dict.keys()):  # use list to take a snapshot of the keys
            if "._orig_mod" in key:
                state_dict[key.replace("._orig_mod", "")] = state_dict.pop(key)
        # allow loading from the PLBeatThis lightning checkpoint
        for key in list(state_dict.keys()):  # use list to take a snapshot of the keys
            if "model." in key:
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        for key in list(state_dict.keys()):  # use list to take a snapshot of the keys
            if "._orig_mod" in key:
                state_dict[key.replace("._orig_mod", "")] = state_dict.pop(key)
        return state_dict



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
    
class PartialFTTrasformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block alternatively across frequencies and across time.

    Returns a tensor of the same shape as the input. 
    """

    def __init__(self, dim, dim_head, n_head, rotary_embed, dropout):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = roformer.Attention(dim, heads=n_head, dim_head=dim_head,
                                       dropout=dropout, rotary_embed=rotary_embed)
        self.ffF = roformer.FeedForward(dim, dropout=dropout)
        # time directed partial transformer
        self.attnT = roformer.Attention(dim, heads=n_head, dim_head=dim_head,
                                       dropout=dropout, rotary_embed=rotary_embed)
        self.ffT = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        # if self.direction == 'f':
        #     pattern = "(b t) f c"
        # elif self.direction == 't':
        #     pattern = "(b f) t c"
        # frequency directed partial transformer
        x = rearrange(x, f"b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, f"(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, f"(b f) t c -> b c f t", b=b)
        return x


class SumHead(nn.Module):
    """
    A PyTorch module that produces the final beat and downbeat prediction logits.
    The beats are a sum of all beats and all downbeats predictions, to reduce the prediction
    of downbeats which are not beats.
    """
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
