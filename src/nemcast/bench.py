"""CPU latency benchmark for Timestral model sizes on NEM-shaped inputs."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from timestral import TimestralForecaster

SIZES = ("5M", "28M", "123M", "209M")
CASES = {
    # name: (batch, variates, context, horizon)
    "DA-30 uni (5x1, ctx 32768, h 79)": (5, 1, 32768, 79),
    "DA-30 multi (1x5, ctx 32768, h 79)": (1, 5, 32768, 79),
    "ID-5 uni (5x1, ctx 32768, h 144)": (5, 1, 32768, 144),
    "ID-5 uni (5x1, ctx 8640, h 144)": (5, 1, 8640, 144),
}


def bench(sizes: list[str], repeats: int) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for size in sizes:
        t0 = time.perf_counter()
        fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")
        load_s = time.perf_counter() - t0
        for case, (b, v, t, h) in CASES.items():
            ctx = rng.standard_normal((b, v, t)).cumsum(-1).astype(np.float32)
            for sym in (True, False):
                with torch.inference_mode():
                    fc.forecast(ctx[..., -512:], prediction_length=h, symmetric_averaging=sym)
                    times = []
                    for _ in range(repeats):
                        s = time.perf_counter()
                        fc.forecast(ctx, prediction_length=h, symmetric_averaging=sym, clip_nonnegative=False)
                        times.append(time.perf_counter() - s)
                row = {"model": size, "case": case, "symmetric": sym, "seconds": round(min(times), 2)}
                rows.append(row | {"load_s": round(load_s, 1)})
                print(json.dumps(row), flush=True)
        del fc
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", default=list(SIZES), choices=SIZES)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", default="data/results/bench.json")
    args = p.parse_args()
    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    rows = bench(args.sizes, args.repeats)
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
