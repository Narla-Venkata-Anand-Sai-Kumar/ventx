"""Bootstrap VenTX-100K from a Chaos-135M checkpoint's trained weights.

Loads Chaos-135M's real state_dict into a freshly-constructed VentxModel and saves it as
this project's own checkpoint format. Weights only -- no optimizer state crosses over
(different codebase, different optimizer object); every VenTX training stage starts its
own fresh Muon/AdamW state.

Bootstraps from the BASE v1 pretrained checkpoint (step 48000), not the narrow SFT/
reasoning checkpoint -- context length is architectural and belongs before task-specific
fine-tuning, per the plan.

    python scripts/import_chaos_weights.py \
        --src C:/chaos-data/checkpoints/chaos135m-v1_step48000.pt \
        --out C:/ventx-data/checkpoints/ventx_stage0.pt
"""
import argparse
import os
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ventx.config import VentxConfig   # noqa: E402
from ventx.model import VentxModel     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Chaos-135M checkpoint .pt")
    ap.add_argument("--out", required=True, help="output path for the VenTX checkpoint")
    ap.add_argument("--max-seq-len", type=int, default=65536)
    ap.add_argument("--rope-theta", type=float, default=362_000.0)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    src_cfg = ck["cfg"]
    print(f"source checkpoint: {a.src}")
    print(f"  step={ck.get('step')}, src max_seq_len={src_cfg.get('max_seq_len')}, "
          f"src rope_theta={src_cfg.get('rope_theta')}")

    # Architecture fields must match VentxConfig's defaults exactly for a clean copy --
    # fail loudly here rather than let load_state_dict silently mismatch shapes.
    must_match = dict(vocab_size=49152, n_layer=12, d_model=768, n_head=12, n_kv_head=4,
                      ffn_hidden=2816, qk_norm=True, tie_embeddings=True)
    mismatches = {k: (src_cfg.get(k), v) for k, v in must_match.items() if src_cfg.get(k) != v}
    if mismatches:
        raise ValueError(f"Source checkpoint architecture doesn't match VentxConfig: {mismatches}")

    vcfg = VentxConfig(max_seq_len=a.max_seq_len, rope_theta=a.rope_theta)
    model = VentxModel(vcfg)

    # strict=True raises RuntimeError on any missing/unexpected key -- if this line
    # doesn't raise, the copy was exact.
    model.load_state_dict(ck["model"], strict=True)
    print("state_dict loaded with strict=True -- no missing/unexpected keys.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(model=model.state_dict(), cfg=asdict(vcfg),
                    step=0, source=a.src, source_step=ck.get("step")), a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
