from einops import repeat, rearrange
import math
import torch
from torch_utils import persistence
import sys
import contextlib
from io import StringIO
import torch.nn.functional as F
import random


@persistence.persistent_class
class FMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def skewed_timestep_sample(self, num_samples, device):
        P_mean = -1.2
        P_std = 1.2
        rnd_normal = torch.randn((num_samples,), device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        time = 1 / (1 + sigma)
        time = torch.clip(time, min=1e-4, max=1.0)
        return time

    def __call__(self, net, images, n_samples, labels=None, augment_pipe=None):
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        batch_num = y.shape[0]
        device = y.device
        t = self.skewed_timestep_sample(batch_num, device=device).view(-1, 1, 1, 1)
        alpha_t = t.view(-1, 1, 1, 1)
        sigma_t = (1 - t).view(-1, 1, 1, 1)
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -torch.ones_like(t)
        n = torch.randn_like(y)
        y_t = sigma_t * n + alpha_t * y
        u_t = d_sigma_t * n + d_alpha_t * y
        flow_loss_v = torch.pow(net(y_t, t) - u_t, 2).mean()
        flow_loss_div = torch.zeros_like(flow_loss_v)
        bootstrap_loss_v = torch.zeros_like(flow_loss_v)
        bootstrap_loss_div = torch.zeros_like(flow_loss_v)

        return flow_loss_v, flow_loss_div, bootstrap_loss_v, bootstrap_loss_div


@persistence.persistent_class
class ShortcutLoss:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        teacher=None,
        denoise_timesteps=1024,
        amp_dtype=torch.float16,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.teacher = teacher
        self.denoise_timesteps = denoise_timesteps
        self.amp_dtype = amp_dtype

    def sample_bootstrap_timesteps(self, B, device):
        dt_choices = torch.tensor([1.0 / (2**i) for i in range(7)], device=device)
        dt_idx = torch.randint(0, len(dt_choices), (B,), device=device)
        dt = dt_choices[dt_idx]
        max_steps = ((1.0 - dt) / dt).floor().long() + 1
        step = (torch.rand(B, device=device) * max_steps.float()).floor()
        t_start = step * dt
        t_end = t_start + dt
        t_mid = (t_start + t_end) / 2
        return (
            dt.view(-1, 1, 1, 1),
            t_start.view(-1, 1, 1, 1),
            t_mid.view(-1, 1, 1, 1),
        )

    def __call__(self, net, images, n_samples, labels=None, augment_pipe=None):
        y, _ = augment_pipe(images) if augment_pipe is not None else (images, None)
        B, device = y.shape[0], y.device

        # ===== Flow loss =====
        t_discrete = torch.randint(0, self.denoise_timesteps, (B,), device=device)
        t = (t_discrete.float() / self.denoise_timesteps).view(-1, 1, 1, 1)

        t = t.clamp(min=1e-9, max=1.0)
        n = torch.randn_like(y)
        x_t = (1 - (1 - 1e-9) * t) * n + t * y
        x_t.requires_grad_(True)

        v_hat, _ = net(x_t, t, torch.ones_like(t) / self.denoise_timesteps)
        v_teacher = self.teacher(x_t, t)

        flow_loss_v = F.mse_loss(v_hat, v_teacher.detach())
        flow_loss_div = torch.zeros_like(flow_loss_v)

        # ===== Bootstrap loss 2 =====
        dt, t_start, t_mid = self.sample_bootstrap_timesteps(B, device)
        t_start = t_start.clamp(min=1e-9, max=1.0)

        x_t_start = (1 - (1 - 1e-9) * t_start) * torch.randn_like(y) + t_start * y
        x_t_start.requires_grad_(True)

        with torch.no_grad():
            v_t_start, _ = net(x_t_start, t_start, dt / 2)
            v_t_mid, _ = net(x_t_start + (t_mid - t_start) * v_t_start, t_mid, dt / 2)
            v_t_target = (v_t_start + v_t_mid) / 2
        v_hat_bootstrap, _ = net(x_t_start, t_start, dt)

        bootstrap_loss_v = F.mse_loss(v_hat_bootstrap, v_t_target.detach())
        bootstrap_loss_div = torch.zeros_like(bootstrap_loss_v)

        return flow_loss_v, flow_loss_div, bootstrap_loss_v, bootstrap_loss_div


@persistence.persistent_class
class LikelihoodLoss:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        teacher=None,
        denoise_timesteps=1024,
        amp_dtype=torch.float16,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.teacher = teacher
        self.denoise_timesteps = denoise_timesteps
        self.amp_dtype = amp_dtype

    def compute_hutchinson_div_from_v(self, v, x):

        assert x.requires_grad, "x must require grad"
        eps = torch.randint(0, 2, (1, *x.shape), device=x.device).float() * 2 - 1
        proj = (v.unsqueeze(0) * eps).sum(dim=tuple(range(2, eps.ndim)))  # [1, B]
        grads = torch.autograd.grad(
            proj.sum(), x, create_graph=True, retain_graph=False
        )[
            0
        ]  # [B,...]

        div = (grads.unsqueeze(0) * eps).sum(dim=tuple(range(2, eps.ndim)))  # [1, B]

        return div.squeeze(0)  # [B]

    def sample_bootstrap_timesteps(self, B, device):
        dt_choices = torch.tensor([1.0 / (2**i) for i in range(7)], device=device)
        dt_idx = torch.randint(0, len(dt_choices), (B,), device=device)
        dt = dt_choices[dt_idx]
        max_steps = ((1.0 - dt) / dt).floor().long() + 1
        step = (torch.rand(B, device=device) * max_steps.float()).floor()
        t_start = step * dt
        t_end = t_start + dt
        t_mid = (t_start + t_end) / 2
        return (
            dt.view(-1, 1, 1, 1),
            t_start.view(-1, 1, 1, 1),
            t_mid.view(-1, 1, 1, 1),
        )

    def __call__(self, net, images, n_samples, labels=None, augment_pipe=None):
        y, _ = augment_pipe(images) if augment_pipe is not None else (images, None)
        B, device = y.shape[0], y.device

        # ===== Flow loss =====
        t_discrete = torch.randint(0, self.denoise_timesteps, (B,), device=device)
        t = (t_discrete.float() / self.denoise_timesteps).view(-1, 1, 1, 1)

        t = t.clamp(min=1e-9, max=1.0)
        n = torch.randn_like(y)
        x_t = (1 - (1 - 1e-9) * t) * n + t * y
        x_t.requires_grad_(True)

        v_hat, div_hat = net(x_t, t, torch.ones_like(t) / self.denoise_timesteps)
        v_teacher = self.teacher(x_t, t)
        div_teacher = (
            self.compute_hutchinson_div_from_v(v_teacher, x_t).detach() / 20000.0
        )

        flow_loss_div = F.mse_loss(div_hat.view(-1), div_teacher.detach())
        flow_loss_v = F.mse_loss(v_hat, v_teacher.detach())

        # ===== Bootstrap loss 2 =====
        dt, t_start, t_mid = self.sample_bootstrap_timesteps(B, device)
        t_start = t_start.clamp(min=1e-9, max=1.0)

        x_t_start = (1 - (1 - 1e-9) * t_start) * torch.randn_like(y) + t_start * y
        x_t_start.requires_grad_(True)

        with torch.no_grad():
            v_t_start, div_t_start = net(x_t_start, t_start, dt / 2)
            v_t_mid, div_t_mid = net(
                x_t_start + (t_mid - t_start) * v_t_start, t_mid, dt / 2
            )
            v_t_target = (v_t_start + v_t_mid) / 2
            div_t_target = (div_t_start + div_t_mid) / 2
        v_hat_bootstrap, div_hat_bootstrap = net(x_t_start, t_start, dt)

        bootstrap_loss_div = F.mse_loss(
            div_hat_bootstrap.view(-1), div_t_target.detach()
        )
        bootstrap_loss_v = F.mse_loss(v_hat_bootstrap, v_t_target.detach())

        return flow_loss_v, flow_loss_div, bootstrap_loss_v, bootstrap_loss_div
