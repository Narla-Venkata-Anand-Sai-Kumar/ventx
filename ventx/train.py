"""VenTX-100K training loop: Muon + WSD schedule + gradient accumulation, staged
context extension.

Two distinct ways to start a run, on purpose (this split was a real gap found in
Chaos-135M's own chaos/train.py -- it only had --resume, forcing awkward workarounds
whenever a run needed to continue from another run's weights without inheriting its
optimizer state or step count):

  --init <ckpt>    weights-only bootstrap. Loads only the state_dict, builds a FRESH
                   Muon/AdamW optimizer and a fresh WSD schedule sized to --total-steps,
                   starts step counting at 0. This is how every new context-length stage
                   begins: Stage A --init's the Stage-0 imported checkpoint, Stage B
                   --init's Stage A's final checkpoint, Stage C --init's Stage B's, etc.
                   Momentum/schedule state from a different stage (different seq_len,
                   different theta, different step budget) has no reason to carry over.
  --resume <ckpt>  full-state resume (model + optimizer + step count) -- for continuing
                   an interrupted run of the SAME stage, not for chaining to a new one.

Building a fresh VentxConfig(max_seq_len=a.seq_len, rope_theta=a.rope_theta) and then
load_state_dict()-ing a previous stage's weights into it is exactly the same pattern
scripts/import_chaos_weights.py and scripts/verify_import.py already proved correct --
the RoPE cos/sin cache is a non-persistent buffer, rebuilt fresh from this stage's own
config, and never conflicts with the state_dict being loaded.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ventx.config import VentxConfig, TrainConfig   # noqa: E402
from ventx.data import ShardDataLoader               # noqa: E402
from ventx.model import VentxModel                   # noqa: E402
from ventx.muon import build_optimizer               # noqa: E402


def ntk_theta(seq_len: int, base_seq: int = 2048, base_theta: float = 10000.0, head_dim: int = 64) -> float:
    """NTK-aware RoPE theta scaling: theta' = theta * (L'/L)^(head_dim/(head_dim-2))."""
    scale = seq_len / base_seq
    return base_theta * scale ** (head_dim / (head_dim - 2))


def wsd_lr(step, total, warmup, decay_frac):
    """Warmup -> stable -> 1-sqrt decay, same design as Chaos-135M's schedule."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total - int(total * decay_frac)
    if step < decay_start:
        return 1.0
    frac = (step - decay_start) / max(1, total - decay_start)
    return max(0.0, 1.0 - math.sqrt(frac))


def _rotate_checkpoints(out_dir, run_name, keep):
    if not keep:
        return
    pat = re.compile(rf"^{re.escape(run_name)}_step(\d+)\.pt$")
    found = []
    for p in glob.glob(os.path.join(out_dir, f"{run_name}_step*.pt")):
        m = pat.match(os.path.basename(p))
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    for _, p in found[:-keep]:
        os.remove(p)


@torch.no_grad()
def evaluate(model, loader, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.next_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.float())
    model.train()
    return torch.stack(losses).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="C:/ventx-data/tokenized")
    ap.add_argument("--out-dir", default="C:/ventx-data/checkpoints")
    ap.add_argument("--run-name", default="ventx-stageA")
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--rope-theta", type=float, default=0.0,
                    help="0 = auto NTK-scaled from --seq-len (see ntk_theta())")
    ap.add_argument("--lr-matrix", type=float, default=0.02)
    ap.add_argument("--lr-embed", type=float, default=0.002)
    ap.add_argument("--adamw-8bit", type=int, default=0)
    ap.add_argument("--compile", type=int, default=0,
                    help="off by default: Stage 0 found torch.compile OOMs with grad "
                         "checkpointing at longer seq_len (32768+); safe to try at 8192")
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--ckpt-group", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-minutes", type=float, default=0, help="wall-clock stop; 0 = no limit")
    ap.add_argument("--init", default="", help="weights-only bootstrap from another stage's checkpoint")
    ap.add_argument("--resume", default="", help="full-state resume of an interrupted run of THIS stage")
    ap.add_argument("--peak-tflops", type=float, default=40.83)
    a = ap.parse_args()

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # auto-tunes the best kernel for our fixed
                                             # (batch, seq_len) shape -- free, since that
                                             # shape never changes within a training stage
    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, f"{a.run_name}.jsonl")

    theta = a.rope_theta if a.rope_theta > 0 else ntk_theta(a.seq_len)
    cfg = VentxConfig(max_seq_len=a.seq_len, rope_theta=theta)
    tcfg = TrainConfig(lr_matrix=a.lr_matrix, lr_embed=a.lr_embed, micro_batch=a.micro_batch,
                       grad_accum=a.grad_accum, seq_len=a.seq_len, total_steps=a.total_steps,
                       warmup_steps=a.warmup_steps, grad_checkpoint=bool(a.ckpt))

    model = VentxModel(cfg).cuda()
    model.train()
    model.enable_grad_checkpoint(bool(a.ckpt), group=a.ckpt_group)

    start_step = 0
    if a.init and os.path.exists(a.init):
        ck = torch.load(a.init, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialized weights from {a.init} (source step={ck.get('step')}) -- "
              f"fresh optimizer, fresh schedule, step 0", flush=True)

    opt = build_optimizer(model, tcfg, adamw_8bit=bool(a.adamw_8bit))
    base_lrs = [g["lr"] for g in opt.param_groups]

    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {a.resume} at step {start_step}", flush=True)

    run = torch.compile(model, dynamic=False) if a.compile else model

    train_loader = ShardDataLoader(a.data_dir, "train", a.micro_batch, a.seq_len)
    val_loader = ShardDataLoader(a.data_dir, "val", a.micro_batch, a.seq_len, seed=99)

    tok_per_step = a.micro_batch * a.grad_accum * a.seq_len
    fpt = model.flops_per_token(a.seq_len)
    print(json.dumps(dict(params_m=round(model.num_params() / 1e6, 2),
                          seq_len=a.seq_len, rope_theta=round(theta, 1),
                          tokens_per_step=tok_per_step,
                          total_tokens=tok_per_step * a.total_steps,
                          train_tokens_available=train_loader.total_tokens,
                          flops_per_token=round(fpt / 1e9, 3)), indent=2), flush=True)

    t_start = time.time()
    t_log = time.time()
    for step in range(start_step, a.total_steps):
        mult = wsd_lr(step, a.total_steps, a.warmup_steps, tcfg.decay_frac)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * mult

        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(a.grad_accum):
            x, y = train_loader.next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = run(x, y)
            (loss / a.grad_accum).backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if (step + 1) % a.log_every == 0:
            torch.cuda.synchronize()
            dt = (time.time() - t_log) / a.log_every
            t_log = time.time()
            tl = (loss_sum / a.grad_accum).item()
            rec = dict(step=step + 1, loss=round(tl, 4), ppl=round(math.exp(min(tl, 20)), 2),
                       lr_mult=round(mult, 4), grad_norm=round(float(gnorm), 3),
                       tok_per_s=round(tok_per_step / dt), step_s=round(dt, 2),
                       mfu_pct=round(100 * tok_per_step * fpt / dt / (a.peak_tflops * 1e12), 1),
                       tokens=(step + 1) * tok_per_step,
                       elapsed_min=round((time.time() - t_start) / 60, 1),
                       vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (step + 1) % a.eval_every == 0:
            vl = evaluate(model, val_loader)
            rec = dict(step=step + 1, val_loss=round(vl, 4), val_ppl=round(math.exp(min(vl, 20)), 2),
                       tokens=(step + 1) * tok_per_step)
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if (step + 1) % a.save_every == 0 or step + 1 == a.total_steps:
            p = os.path.join(a.out_dir, f"{a.run_name}_step{step+1}.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1,
                            cfg=asdict(cfg), tcfg=asdict(tcfg)), p)
            print(f"saved {p}", flush=True)
            _rotate_checkpoints(a.out_dir, a.run_name, a.keep_checkpoints)

        if a.max_minutes and (time.time() - t_start) / 60 >= a.max_minutes:
            p = os.path.join(a.out_dir, f"{a.run_name}_step{step+1}.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1,
                            cfg=asdict(cfg), tcfg=asdict(tcfg)), p)
            print(f"time limit reached; saved {p}", flush=True)
            break


if __name__ == "__main__":
    main()
