"""
Model definitions for the Beat This! beat tracker.
"""

from collections import OrderedDict

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from beat_this.model import roformer
from beat_this.utils import replace_state_dict_key


class BeatThis(nn.Module):
    """
    A neural network model for beat tracking. It is composed of three main components:
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
            (default: {"frontend": 0.1, "transformer": 0.2}).
        sum_head (bool): Whether to use a SumHead for the final predictions (default: True) or plain independent projections.
        partial_transformers (bool): Whether to include partial frequency- and time-transformers in the frontend (default: True)
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
        sum_head: bool = True,
        partial_transformers: bool = True,
    ):
        super().__init__()
        # shared rotary embedding for frontend blocks and transformer blocks
        rotary_embed = RotaryEmbedding(head_dim)

        # create the frontend
        # - stem
        stem = self.make_stem(spect_dim, stem_dim)
        spect_dim //= 4  # frequencies were convolved with stride 4
        # - three frontend blocks
        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                self.make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers,
                    head_dim,
                    rotary_embed,
                    dropout["frontend"],
                )
            )
            dim *= 2
            spect_dim //= 2  # frequencies were convolved with stride 2
        frontend_blocks = nn.Sequential(*frontend_blocks)
        # - linear projection to transformer dimensionality
        concat = Rearrange("b c f t -> b t (c f)")
        linear = nn.Linear(dim * spect_dim, transformer_dim)
        self.frontend = nn.Sequential(
            OrderedDict(stem=stem, blocks=frontend_blocks, concat=concat, linear=linear)
        )

        # create the transformer blocks
        assert (
            transformer_dim % head_dim == 0
        ), "transformer_dim must be divisible by head_dim"
        n_heads = transformer_dim // head_dim
        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )

        # create the output heads
        if sum_head:
            self.task_heads = SumHead(transformer_dim)
        else:
            self.task_heads = Head(transformer_dim)

        # init all weights
        self.apply(self._init_weights)

    @staticmethod
    def make_stem(spect_dim: int, stem_dim: int) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                rearrange_tf=Rearrange("b t f -> b f t"),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(
                    in_channels=1,
                    out_channels=stem_dim,
                    kernel_size=(4, 3),
                    stride=(4, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                bn2d=nn.BatchNorm2d(stem_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        partial_transformers: bool = True,
        head_dim: int | None = 32,
        rotary_embed: RotaryEmbedding | None = None,
        dropout: float = 0.1,
    ) -> nn.Module:
        if partial_transformers and (head_dim is None or rotary_embed is None):
            raise ValueError(
                "Must specify head_dim and rotary_embed for using partial_transformers"
            )
        return nn.Sequential(
            OrderedDict(
                partial=(
                    PartialFTTransformer(
                        dim=in_dim,
                        dim_head=head_dim,
                        n_head=in_dim // head_dim,
                        rotary_embed=rotary_embed,
                        dropout=dropout,
                    )
                    if partial_transformers
                    else nn.Identity()
                ),
                # conv block
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                # out_channels : 64, 128, 256
                # freqs : 16, 8, 4 (due to the stride=2)
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.transformer_blocks(x)
        x = self.task_heads(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class PartialRoformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block either only across frequencies or only across time.
    Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        direction: str,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        self.direction = direction[0].lower()
        if self.direction not in "ft":
            raise ValueError(f"direction must be F or T, got {direction}")
        self.attn = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ff = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        if self.direction == "f":
            pattern = "(b t) f c"
        elif self.direction == "t":
            pattern = "(b f) t c"
        x = rearrange(x, f"b c f t -> {pattern}")
        x = x + self.attn(x)
        x = x + self.ff(x)
        x = rearrange(x, f"{pattern} -> b c f t", b=b)
        return x


class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time. Same
    as applying two PartialRoformer() in sequence, but encapsulated in a single
    module. Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = roformer.FeedForward(dim, dropout=dropout)
        # time directed partial transformer
        self.attnT = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
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
        # autocast to float16 disabled to avoid numerical issues causing NaNs
        with torch.autocast(beat.device.type, enabled=False):
            beat = beat.float() + downbeat.float()
        return {"beat": beat, "downbeat": downbeat}


class Head(nn.Module):
    """
    A PyToch module that produces the final beat and downbeat prediction logits with independent linear layers outputs.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x):
        beat_downbeat = self.beat_downbeat_lin(x)
        # separate beat from downbeat
        beat, downbeat = rearrange(beat_downbeat, "b t c -> c b t", c=2)
        return {"beat": beat, "downbeat": downbeat}
