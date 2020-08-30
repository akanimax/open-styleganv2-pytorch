from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import AvgPool2d, Conv2d, LeakyReLU, Linear, Module, Sequential
from torch.nn.functional import conv2d


class EqualizedConv2d(Conv2d):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return conv2d(
            input=x,
            weight=self.weight * self.scale,  # scale the weight on runtime
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class EqualizedLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_features
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)


class AdaIN(Module):
    """
    AdaIN layer
    Args:
        mod_vector_channels: number of channels in the modulating vector
        num_channels: number of channels in the input_volume
    """

    def __init__(self, mod_vector_channels: int, num_channels: int) -> None:
        super().__init__()
        self.mod_vector_channels = mod_vector_channels
        self.num_channels = num_channels
        self.mu_transformer = EqualizedLinear(mod_vector_channels, num_channels)
        self.sigma_transformer = EqualizedLinear(mod_vector_channels, num_channels)

    def forward(self, x: Tensor, mod_vector: Tensor, alpha=1e-8) -> Tensor:
        """
        applies the AdaIN transform on the input
        Args:
            x: input volume [B x c x h x w]
            mod_vector: vector for obtaining the AdaIN [B x d]
            alpha: a very small value for numerical stability
        Returns: Adaptively instance normalized input
        """
        # instance normalize x:
        x_in = (x - x.mean(dim=(-1, -2), keepdim=True)) / (
            x.std(dim=(-1, -2), keepdim=True) + alpha
        )

        # obtain the new mus and sigmas
        mus = self.mu_transformer(mod_vector)
        sigmas = 1 + self.sigma_transformer(mod_vector)

        # reshape the mus and sigmas properly
        batch_size, num_channels = mus.shape
        mus, sigmas = (
            mus.reshape((batch_size, num_channels, 1, 1)),
            sigmas.reshape((batch_size, num_channels, 1, 1)),
        )

        # apply the mus and sigmas to the instance normalized x
        return (x_in * sigmas) + mus


class NoOp(Module):
    """ Useful for code readability """

    @staticmethod
    def forward(x: Any) -> Any:
        return x


class MinibatchStdDev(Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
        num_new_features: number of additional feature maps added
    """

    def __init__(self, group_size: int = 4, num_new_features: int = 1) -> None:
        """

        Args:
            group_size:
            num_new_features:
        """
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}, num_new_features={self.num_new_features}"

    def forward(self, x: Tensor, alpha: float = 1e-8) -> Tensor:
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        batch_size, channels, height, width = x.shape

        # reshape x and create the splits of the input accordingly
        y = torch.reshape(
            x,
            [
                batch_size,
                self.num_new_features,
                channels // self.num_new_features,
                height,
                width,
            ],
        )

        y_split = y.split(self.group_size)
        y_list: List[Tensor] = []
        for y in y_split:
            group_size = y.shape[0]

            # [G x M x C' x H x W] Subtract mean over batch.
            y = y - y.mean(dim=0, keepdim=True)

            # [G x M x C' x H x W] Calc standard deviation over batch
            y = torch.sqrt(y.square().mean(dim=0, keepdim=False) + alpha)

            # [M x C' x H x W]  Take average over feature_maps and pixels.
            y = y.mean(dim=[1, 2, 3], keepdim=True)

            # [M x 1 x 1 x 1] Split channels into c channel groups
            y = y.mean(dim=1, keepdim=False)

            # [M x 1 x 1]  Replicate over group and pixels.
            y = y.view((1, *y.shape)).repeat(group_size, 1, height, width)

            # append this to the y_list:
            y_list.append(y)

        y = torch.cat(y_list, dim=0)

        # [B x (N + C) x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


class EncoderBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = partial(
            EqualizedConv2d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.normal_path = Sequential(
            conv(in_channels, out_channels),
            LeakyReLU(),
            conv(out_channels, out_channels),
            LeakyReLU(),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.residual_path = Sequential(
            EqualizedConv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)
            ),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.normal_path(x)
        res_y = self.residual_path(x)
        return (y + res_y) * (1 / np.sqrt(2))


class ApplyNoise(Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

        self.weights = torch.nn.Parameter(torch.zeros((1, num_channels, 1, 1)))

    def forward(self, x: Tensor, noise_input: Optional[Tensor] = None) -> Tensor:
        if noise_input is not None:
            assert noise_input.shape[1] == 1, "noise is not single channel"
            assert (
                noise_input.shape[-2:] == x.shape[-2:]
            ), "noise dimensions incompatible"
        if noise_input is None:
            noise_input = torch.randn((x.shape[0], 1, *x.shape[-2:])).to(x.device)
        return x + (noise_input * self.weights)


class DecoderBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_vector_channels: int,
        upsample_input: bool = True,
    ) -> None:
        super().__init__()

        self.mod_vector_channels = mod_vector_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_input = upsample_input

        # modules required by the block:
        conv = partial(
            EqualizedConv2d, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_1 = conv(in_channels, out_channels)
        self.noise_1 = ApplyNoise(out_channels)
        self.conv_2 = conv(out_channels, out_channels)
        self.noise_2 = ApplyNoise(out_channels)
        self.ada_in_1 = AdaIN(mod_vector_channels, out_channels)
        self.ada_in_2 = AdaIN(mod_vector_channels, out_channels)
        self.leaky_relu = LeakyReLU()

    def forward(
        self, x: Tensor, mod_vector: Tensor, noise_input: Optional[Tensor] = None
    ) -> Tensor:
        if self.upsample_input:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv_1(x)
        x = self.noise_1(x, noise_input)
        x = self.leaky_relu(x)
        x = self.ada_in_1(x, mod_vector)
        x = self.conv_2(x)
        x = self.noise_2(x, noise_input)
        x = self.leaky_relu(x)
        x = self.ada_in_2(x, mod_vector)
        return x
