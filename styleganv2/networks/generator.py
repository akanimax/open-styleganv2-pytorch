from functools import partial
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from .shared.custom_layers import DecoderBlock, EqualizedConv2d
from .shared.utils import nf


class Generator(Module):
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

        # object state
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

        self.blocks = ModuleList(
            [
                DecoderBlock(
                    in_channels=self.nf(stage),
                    out_channels=self.nf(stage + 1),
                    mod_vector_channels=latent_size,
                    upsample_input=stage != 1,
                )
                for stage in range(1, depth)
            ]
        )

        self.to_rgbs = ModuleList(
            [
                EqualizedConv2d(self.nf(stage + 1), num_channels, kernel_size=(1, 1))
                for stage in range(1, depth)
            ]
        )
        self.noise_inputs = [
            torch.randn(
                (1, self.nf(stage + 1), (2 ** (stage + 1)), (2 ** (stage + 1)))
            ).requires_grad_(False)
            for stage in range(1, depth)
        ]
        self.initial_block = torch.nn.Parameter(
            torch.randn((1, self.latent_size, 4, 4))
        )

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
            },
            "state_dict": self.state_dict(),
        }

    def forward(
        self, latent_vectors: List[Tensor], randomize_noise: bool = True
    ) -> Tensor:
        assert len(latent_vectors) == self.depth, (
            f"Latent vectors are incompatible with the depth of the network"
            f"len(latent_vectors) = {len(latent_vectors)} depth = {self.depth}"
        )

        ref_latent_vector = latent_vectors[0]
        batch_size = ref_latent_vector.shape[0]
        assert all(
            [
                latent_vector.shape == ref_latent_vector.shape
                for latent_vector in latent_vectors
            ]
        ), f"All latent vectors don't have the same shape :("

        x = self.initial_block.repeat((batch_size, 1, 1, 1))
        y = torch.zeros((batch_size, self.num_channels, 4, 4)).to(x.device)
        for stage, (latent_vector, block, to_rgb, noise_input) in enumerate(
            zip(latent_vectors[1:], self.blocks, self.to_rgbs, self.noise_inputs),
            start=1,
        ):
            x = block(
                x, latent_vector, noise_input=None if randomize_noise else noise_input
            )
            if stage != 1:
                y = torch.nn.functional.interpolate(y, scale_factor=2, mode="bilinear")
            y = y + to_rgb(x)

        return y
