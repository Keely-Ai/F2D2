import math
from einops import repeat, rearrange

import torch
from torch_utils import persistence

from ..networks import (
    SongUNet,
    SongUNet_ctm,
    SongUNet_ctm_scalar,
    DhariwalUNet_ctm_scalar,
    DhariwalUNet,
)


class FM_Net(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.002,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.encoder = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        augment_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        if self.training:
            x = x.squeeze(1)
            B, C, H, W = x.shape
            x = x.to(torch.float32)
            class_labels = (
                None
                if self.label_dim == 0
                else (
                    torch.zeros([1, self.label_dim], device=x.device)
                    if class_labels is None
                    else class_labels.to(torch.float32).reshape(-1, self.label_dim)
                )
            )
            dtype = (
                torch.float16
                if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )

            sigma = sigma.to(torch.float32).reshape(B, 1, 1, 1)
            c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
            c_noise = sigma.log() / 4

            F_x = self.encoder(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                augment_labels=augment_labels,
                **model_kwargs,
            ).view(B, C, H, W)

            return F_x

        else:
            B, C, H, W = x.shape
            x = x.to(torch.float32)
            class_labels = (
                None
                if self.label_dim == 0
                else (
                    torch.zeros([1, self.label_dim], device=x.device)
                    if class_labels is None
                    else class_labels.to(torch.float32).reshape(-1, self.label_dim)
                )
            )
            dtype = (
                torch.float16
                if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )
            # print(f"self.training:{self.training}, sigma:{sigma}, sigma.shape:{sigma.shape}")
            if sigma.numel() == B:
                sigma = sigma.to(torch.float32).reshape(B, 1, 1, 1)
            elif sigma.numel() == 1:
                sigma = sigma.to(torch.float32).reshape(1, 1, 1, 1).repeat(B, 1, 1, 1)
            c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
            c_noise = sigma.log() / 4
            F_x = self.encoder(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                augment_labels=augment_labels,
                **model_kwargs,
            )

            return F_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class ShortCut_Net(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.002,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.encoder = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x,
        sigma_s,
        sigma_t,
        class_labels=None,
        augment_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        if self.training:
            x = x.squeeze(1)
            B, C, H, W = x.shape
            x = x.to(torch.float32)
            sigma_s = sigma_s.to(torch.float32).reshape(B, 1, 1, 1)
            sigma_t = sigma_t.to(torch.float32).reshape(B, 1, 1, 1)

            class_labels = (
                None
                if self.label_dim == 0
                else (
                    torch.zeros([1, self.label_dim], device=x.device)
                    if class_labels is None
                    else class_labels.to(torch.float32).reshape(-1, self.label_dim)
                )
            )
            dtype = (
                torch.float16
                if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )

            c_in = 1 / (sigma_s**2 + self.sigma_data**2) ** 0.5
            c_noise_s = (sigma_s + 1e-9).log() / 4
            c_noise_t = sigma_t.log() / 4

            v, div = self.encoder(
                (c_in * x).to(dtype),
                c_noise_s.flatten(),
                c_noise_t.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )

            return v, div
        else:
            B, C, H, W = x.shape
            x = x.to(torch.float32)
            if sigma_s.numel() == B:
                sigma_s = sigma_s.to(torch.float32).reshape(B, 1, 1, 1)
            elif sigma_s.numel() == 1:
                sigma_s = (
                    sigma_s.to(torch.float32).reshape(1, 1, 1, 1).repeat(B, 1, 1, 1)
                )

            class_labels = (
                None
                if self.label_dim == 0
                else (
                    torch.zeros([1, self.label_dim], device=x.device)
                    if class_labels is None
                    else class_labels.to(torch.float32).reshape(-1, self.label_dim)
                )
            )
            dtype = (
                torch.float16
                if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )

            c_in = 1 / (sigma_s**2 + self.sigma_data**2) ** 0.5
            c_noise_s = (sigma_s + 1e-9).log() / 4
            c_noise_t = sigma_t.log() / 4
            v, div = self.encoder(
                (c_in * x).to(dtype),
                c_noise_s.flatten(),
                c_noise_t.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )

            return v, div
