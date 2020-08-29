from functools import partial
from typing import Any, Dict

from torch import Tensor
from torch.nn import LeakyReLU, Module, ModuleList, Sequential

from .shared.custom_layers import (
    EncoderBlock,
    EqualizedConv2d,
    EqualizedLinear,
    MinibatchStdDev,
    NoOp,
)
from .shared.utils import nf


class Encoder(Module):
    def __init__(
        self,
        depth: int = 8,
        num_channels: int = 3,
        latent_size: int = 512,
        fmap_base: int = 16 << 10,
        fmap_decay: float = 1.0,
        fmap_min: int = 1,
        fmap_max: int = 512,
        use_minbatch_stddev: bool = False,
    ) -> None:
        super().__init__()

        # object state
        self.use_minbatch_stddev = use_minbatch_stddev
        self.depth = depth
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_min = fmap_min
        self.fmap_max = fmap_max

        # construct a shorthand for the nf:
        self.nf = partial(
            nf,
            fmap_base=fmap_base,
            fmap_decay=fmap_decay,
            fmap_max=fmap_max,
            fmap_min=fmap_min,
        )

        # create the encoder blocks:
        self.from_rgb = EqualizedConv2d(
            num_channels, self.nf(depth), kernel_size=(1, 1)
        )
        self.blocks = ModuleList([self.block(stage) for stage in range(depth, 1, -1)])

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "depth": self.depth,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "fmap_base": self.fmap_base,
                "fmap_decay": self.fmap_decay,
                "fmap_min": self.fmap_min,
                "fmap_max": self.fmap_max,
                "use_minbatch_std": self.use_minbatch_stddev,
            },
            "state_dict": self.state_dict(),
        }

    def block(self, stage: int) -> Module:
        if stage >= 3:  # 8x8 resolution and up
            return EncoderBlock(self.nf(stage), self.nf(stage - 1))
        else:  # 4x4 resolution
            first_in_channels = self.nf(stage) + (1 if self.use_minbatch_stddev else 0)
            return Sequential(
                MinibatchStdDev() if self.use_minbatch_stddev else NoOp(),
                EqualizedConv2d(
                    first_in_channels,
                    self.nf(stage - 1),
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
                LeakyReLU(),
                EqualizedConv2d(
                    self.nf(stage - 1), self.nf(stage - 2), kernel_size=(4, 4)
                ),
                LeakyReLU(),
                # 2 * latent_size for the reparameterization trick
                EqualizedConv2d(
                    self.nf(stage - 2), 2 * self.latent_size, kernel_size=(1, 1)
                ),
            )

    def forward(
        self, x: Tensor, alpha: float = 1e-7, normalize_embeddings: bool = True
    ) -> Tensor:
        y = self.from_rgb(x)
        for block in self.blocks:
            y = block(y)
        y = y.squeeze(-1).squeeze(-1)
        if normalize_embeddings:
            # normalize y to fall on unit sphere
            y = y / (y.norm(dim=-1, keepdim=True) + alpha)
        return y


class Discriminator(Module):
    def __init__(
        self,
        depth: int = 8,
        num_channels: int = 3,
        latent_size: int = 512,
        fmap_base: int = 16 << 10,
        fmap_decay: float = 1.0,
        fmap_min: int = 1,
        fmap_max: int = 512,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.depth = depth
        self.encoder = Encoder(
            depth=depth,
            num_channels=num_channels,
            latent_size=latent_size // 2,
            fmap_base=fmap_base,
            fmap_decay=fmap_decay,
            fmap_min=fmap_min,
            fmap_max=fmap_max,
            use_minbatch_stddev=True,
        )
        self.critic = EqualizedLinear(self.latent_size, 1, bias=True)

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {"depth": self.depth, "latent_size": self.latent_size},
            "encoder": self.encoder.get_save_info(),
        }

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.encoder(x, normalize_embeddings=False)
        scores = self.critic(embeddings)
        return scores
