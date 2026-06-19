# v0 vs v1 — Summary

Side-by-side of the two Moving-MNIST 3D-DDPM training runs. Both share the same data and
schedule (`--size 32`, `--frames 10`, `--digits 1`, `--mults 1 2 4`, `--timesteps 1000`,
`--batch 16`, `--length 20000`, `--epochs 30`, `--lr 2e-4`, full 1000-step DDPM sampler) on
an RTX 4060 Laptop (8 GB). v1 changes only the model size, sampling weights, and precision.
See [`v0.md`](v0.md) and [`v1.md`](v1.md) for the full writeups.

## Comparison
| | v0 (baseline) | v1 |
|---|---|---|
| Base channels | 64 | 96 |
| Params | 39.7M | 89.4M (~2.25x) |
| EMA | none (raw weights) | yes, decay 0.999 (sampled from EMA) |
| Precision | standard / fp16 | bf16 autocast (fp16->NaN fix) |
| Final / converged loss | ~0.01-range late epochs (mean ~= 0.0055-0.01) | ~0.0055 epoch-30 mean |
| Checkpoint size | ~152 MB (`checkpoints/ckpt_baseline_v0_e30.pt`) | ~1.4 GB (`checkpoints/ckpt_epoch30.pt`) |
| Training notes | converges by epoch ~3, then flatlines | fp16 diverged to NaN at epoch 9 -> switched to bf16, resumed from epoch 8, finished clean |

Logs live in `logs/` (`train_baseline_v0.log`, `train_v1.log`); per-epoch preview GIFs in
`samples_v0/` (v0) and `samples_v1/` (v1).

## Qualitative verdict
On a same-seed A/B (seed 1234), **v1 strokes are noticeably bolder and cleaner**, while
**v0 strokes are thin and wispy** with more residual noise. Both runs produce recognizable
digits that bounce and reflect off the canvas edges correctly; the difference is in stroke
quality and temporal steadiness, where v1's extra capacity plus EMA weights win clearly.
The MSE curves are nearly indistinguishable, so the loss alone understates the visible gain.

## Comparison artifacts
- 2x4 same-seed grid: `generations/grid_v1_vs_v0_seed1234.gif`
- Per-digit 0-9 v1-vs-v0 grid: `generations/comparison_by_digit.gif`
- Source clips: `generations/v0_seed1234/` and `generations/v1_seed1234/`
