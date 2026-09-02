"""Post-stage evaluation for VenTX-100K.

Two checks, per the plan -- both aimed at telling a real training problem apart from an
architectural limitation:

  1. Per-position validation-loss probe: bucket per-token loss by position range across
     held-out long documents. True dense attention means every position genuinely sees
     everything before it, so a sharp loss cliff at longer positions indicates a training
     problem (not enough long-range signal yet, bad theta for this stage, etc.), not
     something structurally impossible for this architecture.
  2. Needle-in-haystack retrieval: a unique fact planted at a controlled depth in a long
     context, queried at the end. Dense attention's full-cache-always decode behavior is
     correct by construction (no windowed-cache eviction to sidestep), so this runs
     through the real generate() path directly.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ventx.config import VentxConfig          # noqa: E402
from ventx.data import shard_paths, HEADER_BYTES   # noqa: E402
from ventx.model import VentxModel             # noqa: E402
from ventx.generate import generate            # noqa: E402

BUCKETS = [(0, 2048), (2048, 8192), (8192, 32768), (32768, 65536)]

NEEDLE_CODE = "QK7-429-XJ"
NEEDLE_TEXT = f"\nThe secret verification code is {NEEDLE_CODE}.\n"
QUERY_TEXT = "\nQuestion: What is the secret verification code?\nAnswer: The secret verification code is"


@torch.no_grad()
def _hidden(model, idx):
    B, T = idx.shape
    x = model.embed(idx)
    cos, sin = model.rope_cos[:T].to(x.device), model.rope_sin[:T].to(x.device)
    for blk in model.blocks:
        x = blk(x, cos, sin)
    return model.norm_f(x)


@torch.no_grad()
def per_position_loss(model, idx, targets, chunk=2048):
    """Same row-chunked cross-entropy discipline as the model's own training loss head
    (never materializes full fp32 logits for more than `chunk` rows at once), but returns
    the un-reduced per-position losses instead of a scalar."""
    x = _hidden(model, idx)
    B, T, D = x.shape
    xf = x.reshape(-1, D)
    tf = targets.reshape(-1)
    losses = torch.empty(xf.size(0), device=x.device, dtype=torch.float32)
    for i in range(0, xf.size(0), chunk):
        logits = F.linear(xf[i:i + chunk], model.lm_head.weight).float()
        losses[i:i + chunk] = F.cross_entropy(logits, tf[i:i + chunk], reduction="none")
    return losses.view(B, T)


@torch.no_grad()
def position_loss_probe(model, data_dir, seq_len, n_windows=8, device="cuda"):
    """Average per-position loss across n_windows non-overlapping (1, seq_len) windows
    pulled sequentially from the held-out val shard -- whole books, no short-doc mix
    (see tokenize_longdocs.py), so a window is genuinely one continuous long document."""
    paths, meta = shard_paths(data_dir, "val")
    tokens = np.memmap(paths[0], dtype=np.uint16, mode="r", offset=HEADER_BYTES)
    need = seq_len + 1
    n_windows = min(n_windows, max(1, len(tokens) // need))

    buckets = [(lo, min(hi, seq_len)) for lo, hi in BUCKETS if lo < seq_len]
    sums = {b: 0.0 for b in buckets}
    counts = {b: 0 for b in buckets}

    for w in range(n_windows):
        buf = np.asarray(tokens[w * need:(w + 1) * need], dtype=np.int64)
        x = torch.from_numpy(buf[:-1]).view(1, seq_len).to(device)
        y = torch.from_numpy(buf[1:]).view(1, seq_len).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = per_position_loss(model, x, y)
        losses = losses[0].float().cpu().numpy()
        for (lo, hi) in buckets:
            sums[(lo, hi)] += float(losses[lo:hi].sum())
            counts[(lo, hi)] += hi - lo

    return {f"{lo}-{hi}": round(sums[(lo, hi)] / max(1, counts[(lo, hi)]), 4)
            for (lo, hi) in buckets}


@torch.no_grad()
def needle_in_haystack(model, tok, data_dir, seq_len, depths=(0.1, 0.25, 0.5, 0.75, 0.9), device="cuda"):
    paths, meta = shard_paths(data_dir, "val")
    tokens = np.memmap(paths[0], dtype=np.uint16, mode="r", offset=HEADER_BYTES)

    needle_ids = tok.encode(NEEDLE_TEXT).ids
    query_ids = tok.encode(QUERY_TEXT).ids
    filler_len = seq_len - len(needle_ids) - len(query_ids) - 32   # margin for generated tokens
    filler = np.asarray(tokens[:filler_len], dtype=np.int64)

    results = {}
    for depth in depths:
        pos = int(filler_len * depth)
        seq = np.concatenate([filler[:pos], np.array(needle_ids, dtype=np.int64),
                              filler[pos:], np.array(query_ids, dtype=np.int64)])
        idx = torch.from_numpy(seq).view(1, -1).to(device)
        out, _ = generate(model, idx, max_new_tokens=16, temperature=0.0)
        gen_ids = out[0, idx.shape[1]:].tolist()
        gen_text = tok.decode(gen_ids)
        results[f"depth_{depth}"] = dict(found=NEEDLE_CODE in gen_text, generated=gen_text.strip())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="C:/ventx-data/tokenized")
    ap.add_argument("--tokenizer", default="C:/chaos-data/tokenizer/chaos-bpe.json")
    ap.add_argument("--seq-len", type=int, default=0, help="0 = use checkpoint's own max_seq_len")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--skip-needle", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cfg = VentxConfig(**ck["cfg"])
    seq_len = a.seq_len or cfg.max_seq_len
    model = VentxModel(cfg).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"checkpoint: {a.ckpt} (step={ck.get('step')}, max_seq_len={cfg.max_seq_len}, "
          f"rope_theta={cfg.rope_theta})")

    probe = position_loss_probe(model, a.data_dir, seq_len, n_windows=a.windows)
    print("per-position loss by bucket:")
    print(json.dumps(probe, indent=2))

    if not a.skip_needle:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        needle = needle_in_haystack(model, tok, a.data_dir, seq_len)
        print("needle-in-haystack:")
        print(json.dumps(needle, indent=2))


if __name__ == "__main__":
    main()
