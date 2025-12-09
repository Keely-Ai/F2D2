import os, pickle, math
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchdiffeq import odeint as odeint
import dnnlib


class CIFAR10TestBatch(Dataset):
    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"].reshape(-1, 3, 32, 32)
        self.images = torch.tensor(data, dtype=torch.uint8)
        self.labels = torch.tensor(entry["labels"], dtype=torch.long)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        return img, self.labels[idx]


# ========== Divergence Estimation ==========
def hutchinson_divergence(u, x, rademacher=True):
    dims = tuple(range(1, x.dim()))
    div_est = 0.0
    eps = (torch.randint_like(x, low=0, high=2, dtype=torch.int64) * 2 - 1).to(x.dtype)
    vjp = torch.autograd.grad(
        (u * eps).sum(), x,
        create_graph=False, retain_graph=True, allow_unused=False
    )[0]
    div_est = div_est + (vjp * eps).sum(dim=dims)
    return div_est


# ========== ODE RHS ==========
def make_ode_rhs(net, time_eval_steps: int, hutch_samples: int, use_rademacher: bool, verbose_rhs: bool):
    def ode_rhs(t_scalar, state):
        x, logp = state
        B = x.shape[0]
        x_req = x.detach().requires_grad_(True)
        t_in = torch.full((B,), float(t_scalar), device=x_req.device, dtype=x_req.dtype)
        dt_in = torch.full((B,), 1.0 / time_eval_steps, device=x_req.device, dtype=x_req.dtype)
        t_in = t_in.clamp(1e-9, 1.0)

        u, div = net(x_req, t_in, dt_in)
        # div = hutchinson_divergence(u, x_req, n_samples=hutch_samples, rademacher=use_rademacher)
        dxdt = u.detach()
        dlogpdt = div.detach() * 20000.0

        if verbose_rhs:
            print(f"t_scalar:{t_scalar}, dlogpdt: {dlogpdt.mean().item()}")

        return (dxdt, dlogpdt)

    return ode_rhs


# ========== Compute NLL ==========
def compute_nll(x_batch, net, time_eval_steps: int, log2_const: torch.Tensor,
                hutch_samples: int, use_rademacher: bool, verbose_rhs: bool):
    device = x_batch.device
    B, d = x_batch.shape[0], x_batch[0].numel()

    U = torch.rand_like(x_batch)
    y = (x_batch * 255.0 + U) / 256.0
    y = y.clamp(0.0, 1.0)
    z = 2.0 * y - 1.0

    log_det_transform = d * torch.log(torch.tensor(2.0, device=device, dtype=z.dtype))
    logp = torch.zeros(B, device=device, dtype=z.dtype)

    t_span = torch.linspace(1.0, 0.0, time_eval_steps + 1, device=device, dtype=z.dtype)
    method = "euler"

    z_T, logp_T = odeint(
        func=make_ode_rhs(net, time_eval_steps, hutch_samples, use_rademacher, verbose_rhs),
        y0=(z, logp),
        t=t_span,
        method=method,
    )

    z0, logp0 = z_T[-1], logp_T[-1]
    log_prior = -0.5 * (z0**2).view(B, -1).sum(-1)
    log_prior = log_prior - 0.5 * d * torch.log(torch.tensor(2 * math.pi, device=device, dtype=z.dtype))
    total_logp = logp0 + log_prior + log_det_transform

    dequant_correction = d * torch.log(torch.tensor(256.0, device=device, dtype=z.dtype))
    total_logp = total_logp - dequant_correction

    bpd = -(total_logp / (d * log2_const.to(device=device, dtype=z.dtype))).mean().item()
    avg_logp = total_logp.mean().item()
    return bpd, avg_logp


def parse_args():
    p = argparse.ArgumentParser("CIFAR-10 NLL eval (Euler)")

    p.add_argument("--test_batch_path", type=str, required=True,
                   help="Path to CIFAR-10 test_batch (pickle).")
    p.add_argument("--model_pkl", type=str, required=True,
                   help="Path/URL to network-snapshot-*.pkl containing key 'ema'.")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--hutch_samples", type=int, default=2)
    p.add_argument("--use_rademacher", type=lambda x: x.lower() in ("1", "true", "yes", "y"), default=True)
    p.add_argument("--time_eval_steps", type=int, default=8)

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=2)

    p.add_argument("--verbose_rhs", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(True)

    device = args.device
    log2_const = torch.log(torch.tensor(2.0, device=device, dtype=torch.float32))

    # DataLoader
    loader = DataLoader(
        CIFAR10TestBatch(args.test_batch_path),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    # Load model
    print(f'Loading model from "{args.model_pkl}"...')
    with dnnlib.util.open_url(args.model_pkl, verbose=True) as f:
        data = pickle.load(f)
    assert "ema" in data, "Pickle does not contain key 'ema'."
    net = data["ema"].to(device).eval()

    torch.backends.cudnn.benchmark = True
    print("Model loaded.")

    # Eval
    nlls = []
    for i, (x_cpu, _) in enumerate(tqdm(loader, desc="Evaluating CIFAR-10 NLL (euler)")):
        x = x_cpu.to(device, non_blocking=True)
        bpd, p = compute_nll(
            x, net,
            time_eval_steps=args.time_eval_steps,
            log2_const=log2_const,
            hutch_samples=args.hutch_samples,
            use_rademacher=args.use_rademacher,
            verbose_rhs=args.verbose_rhs,
        )
        print(f"[Batch {i:03d}] bpd = {bpd:.4f} logp = {p:.4f}", flush=True)
        nlls.append(bpd)

    avg_bpd = sum(nlls) / max(len(nlls), 1)
    print(f"Average BPD over test set: {avg_bpd:.4f}", flush=True)


if __name__ == "__main__":
    main()
