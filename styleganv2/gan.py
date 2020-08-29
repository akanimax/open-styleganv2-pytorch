""" A generic implementation of GAN which is agnostic of what architecture is used for
the generator and discriminator """
import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.nn import DataParallel, Module
from torch.nn.functional import softplus
from torch.utils.tensorboard import SummaryWriter

from .data.multi_image_loader import (
    ImageDirectoryDataset,
    get_data_loader,
    get_transform,
)
from .networks.shared.utils import update_average
from .utils.image_utils import adjust_dynamic_range


class GAN(object):
    def __init__(
        self,
        generator: Module,
        discriminator: Module,
        input_data_range: Tuple[int, int] = (0, 1),
        output_data_range: Tuple[int, int] = (-1, 1),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        assert (
            generator.latent_size == discriminator.latent_size
            and generator.depth == discriminator.depth
        ), "Encoder and Decoder are incompatible"

        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_size = generator.latent_size
        self.input_data_range = input_data_range
        self.output_data_range = output_data_range
        self.depth = generator.depth
        self.latent_distribution = torch.randn

        # wrap the generator and the discriminator networks in DataParallel
        if self.device != torch.device("cpu"):
            self.generator = DataParallel(self.generator).to(self.device)
            self.discriminator = DataParallel(self.discriminator).to(self.device)

        # by default the encoder and the decoder are in eval mode
        self._toggle_all_networks("eval")

        # create an EMA copy of the generator:
        self.generator_shadow = copy.deepcopy(self.generator).to(self.device)

        # initialize the gen_shadow weights equal to the
        # weights of gen
        update_average(self.generator_shadow, self.generator, beta=0)

    def _toggle_all_networks(self, mode="train"):
        for network in (self.generator, self.discriminator):
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    def generate_sample(self) -> Tensor:
        latent = self.latent_distribution((1, self.latent_size)).to(self.device)
        sample = self.generator([latent for _ in range(self.depth)])
        return sample.detach().cpu()

    def _save_image_grid(self, images: Tensor, img_file: Path) -> None:
        normalized_images = adjust_dynamic_range(
            images.cpu(),
            drange_in=self.output_data_range,
            drange_out=(0, 1),
            slack=True,
        )
        torchvision.utils.save_image(
            normalized_images,
            img_file,
            nrow=int(np.ceil(np.sqrt(len(images)))),
            padding=0,
        )

    def _r1_penalty(self, real: Tensor) -> Tensor:
        # forward pass
        real_scores = self.discriminator(real.requires_grad_(True))

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=[torch.sum(real_scores)],
            inputs=[real],
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_penalty = torch.sum(gradient ** 2, dim=-1)
        return gradient_penalty

    def generator_loss(self, fake: Tensor) -> Tensor:
        fake_scores = self.discriminator(fake)
        return softplus(-fake_scores).mean()

    def discriminator_loss(
        self, fake: Tensor, real: Tensor, r1_gamma: float = 10.0
    ) -> Tensor:
        fake_scores = self.discriminator(fake)
        real_scores = self.discriminator(real)
        simple_gan_loss = softplus(fake_scores) + softplus(-real_scores)
        r1_penalty = (r1_gamma * 0.5) * self._r1_penalty(real)
        return (simple_gan_loss + r1_penalty).mean()

    def _generator_mixing_regularization_loss(self, latents: Tensor) -> Tensor:
        latents_list = [latents for _ in range(self.depth)]
        random_mix_point = np.random.randint(1, len(latents_list))
        second_latents = self.normalize(
            self.latent_distribution((latents.shape[0], self.latent_size)).to(
                self.device
            )
        )
        for replace_index in range(random_mix_point, len(latents_list)):
            latents_list[replace_index] = second_latents
        mix_latents_fake_samples = self.generator(latents_list)
        return self.generator_loss(mix_latents_fake_samples)

    def get_save_info(self):
        if self.device == torch.device("cpu"):
            generator_save_info = self.generator.get_save_info()
            discriminator_save_info = self.generator.get_save_info()
            generator_shadow_save_info = self.generator_shadow.get_save_info()
        else:
            generator_save_info = self.generator.module.get_save_info()
            discriminator_save_info = self.generator.module.get_save_info()
            generator_shadow_save_info = self.generator_shadow.module.get_save_info()
        return {
            "generator": generator_save_info,
            "discriminator": discriminator_save_info,
            "generator_shadow": generator_shadow_save_info,
            "input_data_range": self.input_data_range,
            "output_data_range": self.output_data_range,
        }

    @staticmethod
    def _check_grad_ok(network: Module) -> bool:
        grad_ok = True
        for _, param in network.named_parameters():
            if param.grad is not None:
                param_ok = (
                    torch.sum(torch.isnan(param.grad)) == 0
                    and torch.sum(torch.isinf(param.grad)) == 0
                )
                if not param_ok:
                    grad_ok = False
                    break
        return grad_ok

    @staticmethod
    def normalize(x: Tensor, epsilon: float = 1e-8) -> Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)

    def train(
        self,
        train_path: Path,
        rec_dir: bool = True,
        batch_size: int = 2,
        num_workers: int = 3,
        num_epochs: int = 1000,
        gen_learning_rate: float = 0.003,
        dis_learning_rate: float = 0.003,
        mixing_regularization_prob: Optional[float] = 0.9,
        ema_smoothing: float = 10.0,
        per_epoch_num_logs: int = 100,
        checkpoint_freq: int = 100,
        num_batch_repeats: int = 4,
        output_dir: Path = Path("./train"),
    ) -> None:

        # turn encoder and decoder in training mode:
        self._toggle_all_networks("train")

        # setup the training data:
        train_data = get_data_loader(
            ImageDirectoryDataset(
                train_path,
                input_data_range=self.input_data_range,
                output_data_range=self.output_data_range,
                transform=get_transform(
                    new_size=(2 ** self.depth, 2 ** self.depth), flip_horizontal=True
                ),
                rec_dir=rec_dir,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # create the generator and discriminator optimizers
        gen_optim = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=gen_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )
        dis_optim = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=dis_learning_rate,
            betas=(0, 0.99),
            eps=1e-8,
        )

        # verbose stuff
        model_dir, log_dir = output_dir / "models", output_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        real_images_for_render = next(iter(train_data))
        fixed_latents_for_debugging = self.latent_distribution(
            (batch_size, self.latent_size)
        ).to(self.device)
        fixed_latents_for_debugging = [
            fixed_latents_for_debugging for _ in range(self.depth)
        ]
        self._save_image_grid(real_images_for_render, log_dir / "real_images.png")
        self._save_image_grid(
            self.generator_shadow(fixed_latents_for_debugging).detach(),
            log_dir / "fake_images_0.png",
        )

        # tensorboard summarywriter:
        summary = SummaryWriter(str(log_dir / "tensorboard"))

        # training loop
        global_step = 0
        print("beginning training ... ")
        for epoch in range(1, num_epochs + 1):
            print(f"\ncurrent epoch: {epoch}")
            limit = int(np.ceil(len(train_data.dataset) / batch_size))
            for step, data_batch in enumerate(train_data, start=1):
                real = data_batch.to(self.device)

                dis_loss, gen_loss = 0, 0
                dis_overflow_count, gen_overflow_count = 0, 0
                for _ in range(num_batch_repeats):
                    # discriminator optimization
                    latents = self.normalize(
                        self.latent_distribution((batch_size, self.latent_size)).to(
                            self.device
                        )
                    )
                    fake = self.generator([latents for _ in range(self.depth)])

                    d_loss = self.discriminator_loss(fake.detach(), real)
                    dis_optim.zero_grad()
                    d_loss.backward()
                    if self._check_grad_ok(self.discriminator):
                        dis_optim.step()
                    else:
                        dis_overflow_count += 1

                    # generator optimization
                    latents = self.normalize(
                        self.latent_distribution((batch_size, self.latent_size)).to(
                            self.device
                        )
                    )
                    if (
                        mixing_regularization_prob is not None
                        and np.random.uniform() < mixing_regularization_prob
                    ):
                        g_loss = self._generator_mixing_regularization_loss(latents)
                    else:
                        fake = self.generator([latents for _ in range(self.depth)])
                        g_loss = self.generator_loss(fake)

                    gen_optim.zero_grad()
                    g_loss.backward()
                    if self._check_grad_ok(self.generator):
                        gen_optim.step()
                    else:
                        gen_overflow_count += 1

                    # update the shadow generator:
                    update_average(
                        self.generator_shadow,
                        self.generator,
                        # no idea where this heuristic came from, but I am just
                        # using it :D.
                        beta=(0.5 ** (batch_size / (ema_smoothing * 1000))),
                    )

                    dis_loss += d_loss.item()
                    gen_loss += g_loss.item()

                gen_loss /= num_batch_repeats
                dis_loss /= num_batch_repeats
                global_step += 1

                if step % (int(limit / per_epoch_num_logs) + 1) == 0 or step == 1:
                    print(
                        f"current step: {step} \t gen loss: {gen_loss}  "
                        f"dis loss: {dis_loss} "
                    )
                    summary.add_scalar("gen_loss", gen_loss, global_step=global_step)
                    summary.add_scalar("dis_loss", dis_loss, global_step=global_step)
                    summary.add_scalar(
                        "gen_overflow", gen_overflow_count, global_step=global_step
                    )
                    summary.add_scalar(
                        "dis_overflow", dis_overflow_count, global_step=global_step
                    )

                    save_image_path = log_dir / f"fake_images_{global_step}.png"
                    print(f"saving generated images at: {save_image_path}")
                    self._save_image_grid(
                        self.generator_shadow(fixed_latents_for_debugging).detach(),
                        img_file=save_image_path,
                    )

            if epoch % checkpoint_freq == 0:
                torch.save(self.get_save_info(), model_dir / f"model_{epoch}.pth")

        print("training complete ...")
        # restore the networks to the eval mode
        self._toggle_all_networks("eval")
