"""Stage 0 -- measure ONE (seq_len, micro_batch, grad_accum) config and print one JSON
line. Run per-config in its own process: a CUDA OOM leaves the allocator fragmented and
every later measurement in the same process reads low or spuriously OOMs.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ventx.config import VentxConfig, TrainConfig   # noqa: E402
from ventx.model import VentxModel                   # noqa: E402
from ventx.muon import build_optimizer               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--ckpt-group", type=int, default=1)
    ap.add_argument("--adamw-8bit", type=int, default=0)
    ap.add_argument("--no-chunk", type=int, default=0,
                    help="raise FFN/loss chunk size past seq_len, skipping the internal "
                         "checkpoint()-based chunking entirely (only sane on big-VRAM GPUs)")
    ap.add_argument("--peak-tflops", type=float, default=40.83)
    ap.add_argument("--iters", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.92)
    cfg = VentxConfig(max_seq_len=a.seq)

    out = dict(seq=a.seq, batch=a.batch, grad_accum=a.grad_accum,
              compiled=bool(a.compile), ckpt=bool(a.ckpt))
    try:
        model = VentxModel(cfg).cuda()
        model.train()
        model.enable_grad_checkpoint(bool(a.ckpt), group=a.ckpt_group)
        if a.no_chunk:
            model.set_chunk_sizes(ffn_chunk=a.seq + 1, loss_chunk=a.seq + 1)
        fpt = model.flops_per_token(a.seq)
        out["params_m"] = round(model.num_params() / 1e6, 2)
        out["gflop_per_token"] = round(fpt / 1e9, 3)
        opt = build_optimizer(model, TrainConfig(), adamw_8bit=bool(a.adamw_8bit))
        run = torch.compile(model, dynamic=False) if a.compile else model

        x = torch.randint(0, cfg.vocab_size, (a.batch, a.seq), device="cuda")

        def step():
            for _ in range(a.grad_accum):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = run(x, x)
                (loss / a.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            return loss

        warmup_n = 3 if a.compile else 1
        for _ in range(warmup_n):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            loss = step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.iters

        tps = a.batch * a.seq * a.grad_accum / dt
        out.update(ok=True, step_ms=round(dt * 1000, 1), tokens_per_s=round(tps),
                  mfu_pct=round(100 * tps * fpt / (a.peak_tflops * 1e12), 1),
                  peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  loss=round(float(loss), 4))
    except torch.OutOfMemoryError:
        out.update(ok=False, reason="oom",
                  peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    except Exception as e:
        out.update(ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}")
    print("@@JSON@@" + json.dumps(out))


if __name__ == "__main__":
    main()
