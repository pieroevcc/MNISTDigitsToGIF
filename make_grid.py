"""Stitch gen_*.gif clips into a single side-by-side grid GIF."""
import argparse, math
import numpy as np
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["gen_0.gif", "gen_1.gif", "gen_2.gif", "gen_3.gif"])
    p.add_argument("--out", default="grid.gif")
    p.add_argument("--cols", type=int, default=0, help="0 = auto (square-ish)")
    p.add_argument("--pad", type=int, default=2, help="separator pixels between tiles")
    p.add_argument("--scale", type=int, default=4, help="nearest-neighbor upscale factor")
    p.add_argument("--duration", type=float, default=0.12)
    a = p.parse_args()

    # read each gif as [T, H, W] (grayscale)
    clips = []
    for path in a.inputs:
        frames = imageio.mimread(path)
        arr = np.stack([np.asarray(f) for f in frames])      # [T, H, W] or [T, H, W, C]
        if arr.ndim == 4:
            arr = arr[..., 0]                                # GIFs here are grayscale
        clips.append(arr)

    n = len(clips)
    T = min(c.shape[0] for c in clips)                       # align on shortest clip
    H, W = clips[0].shape[1:3]
    cols = a.cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    pad, bg = a.pad, 32                                      # dark-gray gridlines
    grid_h = rows * H + (rows - 1) * pad
    grid_w = cols * W + (cols - 1) * pad

    out_frames = []
    for t in range(T):
        canvas = np.full((grid_h, grid_w), bg, dtype=np.uint8)
        for i, clip in enumerate(clips):
            r, cc = divmod(i, cols)
            y, x = r * (H + pad), cc * (W + pad)
            canvas[y:y + H, x:x + W] = clip[t]
        if a.scale > 1:
            canvas = np.repeat(np.repeat(canvas, a.scale, 0), a.scale, 1)
        out_frames.append(canvas)

    imageio.mimsave(a.out, out_frames, duration=a.duration)
    print(f"wrote {a.out}: {rows}x{cols} grid, {T} frames, "
          f"{out_frames[0].shape[1]}x{out_frames[0].shape[0]} px")


if __name__ == "__main__":
    main()
