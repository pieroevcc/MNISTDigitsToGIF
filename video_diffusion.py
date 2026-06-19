"""
video_diffusion.py
==================
A minimal, from-scratch *video* diffusion model (DDPM) you can train on a single
8 GB GPU (e.g. an RTX 4060 Laptop). Nothing is hidden inside a pretrained model:
the noising process, the denoising network (a 3D U-Net), and the sampling loop
are all implemented by hand.

Data:   synthetic "Moving MNIST" — bouncing digits generated on the fly.
Model:  pixel-space 3D U-Net that predicts the noise added to a clip.
Train:  standard DDPM epsilon-prediction objective (Ho et al. 2020 / 2022).

Run:
    python video_diffusion.py train          # trains, saves checkpoints + sample GIFs
    python video_diffusion.py sample --ckpt ckpt_epoch10.pt   # generate from a checkpoint

Defaults are sized to fit comfortably in 8 GB. Scale up (size/frames/base/digits)
once you confirm it runs.
"""
import argparse, math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================== DATA ==============================
class MovingMNIST(Dataset):
    """Synthesizes short clips of bouncing MNIST digits on the fly.

    Returns tensors of shape [1, T, H, W] normalized to [-1, 1].
    This is the classic toy benchmark for video prediction/generation:
    it has real temporal structure (motion) but is tiny enough to learn fast.
    """
    def __init__(self, root, length=20000, frames=10, size=32, num_digits=1):
        self.length, self.frames, self.size, self.num_digits = length, frames, size, num_digits
        self.ds = size // 2                       # each digit is half the canvas
        from torchvision import datasets          # lazy import (only needed for data)
        mnist = datasets.MNIST(root, train=True, download=True)
        d = mnist.data.float() / 255.0            # [N, 28, 28] in [0,1]
        d = F.interpolate(d[:, None], size=(self.ds, self.ds),
                          mode="bilinear", align_corners=False)[:, 0]
        self.digits = d.numpy()                   # [N, ds, ds]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        T, S, ds = self.frames, self.size, self.ds
        canvas = np.zeros((T, S, S), dtype=np.float32)
        for _ in range(self.num_digits):
            img = self.digits[random.randrange(len(self.digits))]
            pos = np.array([random.uniform(0, S - ds), random.uniform(0, S - ds)])
            vel = np.random.uniform(-3, 3, size=2)
            for t in range(T):
                y, x = int(pos[0]), int(pos[1])
                canvas[t, y:y+ds, x:x+ds] = np.maximum(canvas[t, y:y+ds, x:x+ds], img)
                pos += vel
                for k in (0, 1):                  # reflect off the walls
                    if pos[k] < 0:
                        pos[k] = -pos[k]; vel[k] = -vel[k]
                    if pos[k] > S - ds:
                        pos[k] = 2 * (S - ds) - pos[k]; vel[k] = -vel[k]
        clip = torch.from_numpy(canvas)[None]     # [1, T, S, S]
        return clip * 2 - 1                        # -> [-1, 1]


# ============================== MODEL ==============================
def timestep_embedding(t, dim):
    """Sinusoidal embedding of the diffusion timestep (like positional encodings)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    """Conv3d residual block with timestep conditioning injected as a bias."""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    """Downsample spatially only (keep the time axis intact)."""
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, stride=(1, 2, 2), padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    """Upsample spatially only."""
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.op(x)


class UNet3D(nn.Module):
    """3D U-Net that predicts the noise epsilon given a noisy clip x_t and timestep t."""
    def __init__(self, in_ch=1, base=64, mults=(1, 2, 4), num_res=2):
        super().__init__()
        self.base = base
        t_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(),
                                      nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv3d(in_ch, base, 3, padding=1)

        chs = [base]                              # track channels for skip connections
        ch = base
        self.downs = nn.ModuleList()
        for i, m in enumerate(mults):
            out = base * m
            for _ in range(num_res):
                self.downs.append(ResBlock3D(ch, out, t_dim)); ch = out; chs.append(ch)
            if i != len(mults) - 1:
                self.downs.append(Down(ch)); chs.append(ch)

        self.mid1 = ResBlock3D(ch, ch, t_dim)
        self.mid2 = ResBlock3D(ch, ch, t_dim)

        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            out = base * m
            for _ in range(num_res + 1):
                self.ups.append(ResBlock3D(ch + chs.pop(), out, t_dim)); ch = out
            if i != 0:
                self.ups.append(Up(ch))

        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                 nn.Conv3d(ch, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.in_conv(x)
        hs = [h]
        for m in self.downs:
            h = m(h, temb) if isinstance(m, ResBlock3D) else m(h)
            hs.append(h)
        h = self.mid2(self.mid1(h, temb), temb)
        for m in self.ups:
            if isinstance(m, ResBlock3D):
                h = m(torch.cat([h, hs.pop()], dim=1), temb)
            else:
                h = m(h)
        return self.out(h)


# ============================== DIFFUSION ==============================
class Diffusion:
    """DDPM forward (noising) and reverse (sampling) processes."""
    def __init__(self, timesteps=1000, device="cpu"):
        self.T = timesteps
        beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.beta = beta
        self.alpha = 1 - beta
        self.abar = torch.cumprod(self.alpha, 0)
        self.sqrt_abar = self.abar.sqrt()
        self.sqrt_1m_abar = (1 - self.abar).sqrt()

    def q_sample(self, x0, t, noise):
        """Add t steps of noise to x0 in closed form: x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
        a = self.sqrt_abar[t][:, None, None, None, None]
        b = self.sqrt_1m_abar[t][:, None, None, None, None]
        return a * x0 + b * noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def sample(self, model, shape, device):
        """Start from pure noise and iteratively denoise to a clean clip."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred = model(x, t)
            mean = (x - self.beta[i] / self.sqrt_1m_abar[i] * pred) / self.alpha[i].sqrt()
            x = mean + (self.beta[i].sqrt() * torch.randn_like(x) if i > 0 else 0)
        return x


# ============================== EMA ==============================
class EMA:
    """Exponential moving average of model weights.

    Sampling from an EMA of the trained weights is a standard DDPM trick: it
    averages out the late-training parameter jitter and reliably yields cleaner,
    less noisy samples than the raw weights — for free (no extra training).
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)            # ints/bools: just track latest

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow


# ============================== UTILS ==============================
def to_gif(clip, path):
    """clip: [1, T, H, W] in [-1,1] -> animated GIF."""
    import imageio.v2 as imageio
    v = (((clip.clamp(-1, 1) + 1) / 2) * 255).byte().cpu().numpy()[0]   # [T,H,W]
    imageio.mimsave(path, [v[t] for t in range(v.shape[0])], duration=0.12)


# ============================== TRAIN / SAMPLE ==============================
CKPT_DIR = "checkpoints"                          # all .pt checkpoints live here


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(CKPT_DIR, exist_ok=True)
    ds = MovingMNIST(a.data, length=a.length, frames=a.frames, size=a.size, num_digits=a.digits)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    model = UNet3D(in_ch=1, base=a.base, mults=tuple(a.mults)).to(device)
    diff = Diffusion(a.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    # bf16 autocast: wide exponent range so big-model activations can't overflow to NaN
    # (fp16 did, on base 96). bf16 needs no loss scaling, so the GradScaler stays disabled.
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = EMA(model, decay=a.ema_decay) if a.ema else None
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {nparams:.1f}M | base: {a.base} | ema: {a.ema} (decay {a.ema_decay})")

    start_epoch, step = 1, 0
    if a.resume:                                  # continue a previous run exactly where it stopped
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if ema and "ema" in ck:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0) + 1      # the saved epoch is fully done; start the next one
        step = ck.get("step", 0)
        print(f"resumed from {a.resume}: continuing at epoch {start_epoch} (step {step})")

    for epoch in range(start_epoch, a.epochs + 1):
        for x0 in dl:
            x0 = x0.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                loss = diff.loss(model, x0)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against loss spikes
            scaler.step(opt); scaler.update()
            if ema:
                ema.update(model)
            step += 1
            if step % a.log == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "step": step}
        if ema:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, os.path.join(CKPT_DIR, f"ckpt_epoch{epoch}.pt"))
        torch.save(ckpt, os.path.join(CKPT_DIR, "ckpt_latest.pt"))   # rolling pointer for --resume checkpoints/ckpt_latest.pt
        model.eval()
        if ema:                                   # sample from the EMA weights
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        s = diff.sample(model, (4, 1, a.frames, a.size, a.size), device)
        for j in range(4):
            to_gif(s[j], f"sample_e{epoch}_{j}.gif")
        if ema:
            model.load_state_dict(backup)
        model.train()
        print(f"saved ckpt_epoch{epoch}.pt + sample GIFs")


def sample(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(in_ch=1, base=a.base, mults=tuple(a.mults)).to(device)
    ckpt = torch.load(a.ckpt, map_location=device)
    if isinstance(ckpt, dict) and ("ema" in ckpt or "model" in ckpt):
        state = ckpt["ema"] if (a.use_ema and "ema" in ckpt) else ckpt["model"]
        which = "ema" if (a.use_ema and "ema" in ckpt) else "model"
    else:                                          # legacy raw state_dict (e.g. baseline v0)
        state, which = ckpt, "raw"
    model.load_state_dict(state)
    print(f"loaded {a.ckpt} [{which} weights]")
    model.eval()
    diff = Diffusion(a.timesteps, device=device)
    if a.seed is not None:                          # seed right before sampling so the noise
        random.seed(a.seed); np.random.seed(a.seed)  # is identical across models regardless of
        torch.manual_seed(a.seed)                    # how much RNG weight-init consumed
        if device == "cuda":
            torch.cuda.manual_seed_all(a.seed)
        print(f"seeded sampling noise with {a.seed}")
    s = diff.sample(model, (a.n, 1, a.frames, a.size, a.size), device)
    for j in range(a.n):
        to_gif(s[j], f"gen_{j}.gif")
    print(f"wrote {a.n} GIFs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "sample"])
    p.add_argument("--data", default="./mnist")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--digits", type=int, default=1)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--length", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log", type=int, default=50)
    p.add_argument("--ckpt", default="checkpoints/ckpt_epoch30.pt")
    p.add_argument("--resume", default=None, help="path to a checkpoint to continue training from")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sampling noise")
    # EMA (train): keep an exponential moving average of weights to sample from.
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.999)
    # sampling: prefer EMA weights from the checkpoint when present.
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    a = p.parse_args()
    (train if a.mode == "train" else sample)(a)