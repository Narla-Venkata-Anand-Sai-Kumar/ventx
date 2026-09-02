"""Mandatory logit-match test: confirm the freshly-imported VenTX model actually computes
the same thing as the original Chaos-135M checkpoint, not just that load_state_dict didn't
raise. Runs both models on the same real tokens at the same (2048, native) length and
compares logits directly.
"""
import argparse
import os
import sys

import torch

CHAOS_ROOT = "c:/Users/venka/OneDrive/Desktop/Choas-135M"
VENTX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaos-ckpt", default="C:/chaos-data/checkpoints/chaos135m-v1_step48000.pt")
    ap.add_argument("--ventx-ckpt", default="C:/ventx-data/checkpoints/ventx_stage0.pt")
    ap.add_argument("--seq-len", type=int, default=2048)
    a = ap.parse_args()

    tok_ids = list(range(100, 100 + a.seq_len))  # deterministic, real-token-range input
    idx = torch.tensor([tok_ids])

    sys.path.insert(0, CHAOS_ROOT)
    from chaos.config import ChaosConfig
    from chaos.model import ChaosModel
    ck = torch.load(a.chaos_ckpt, map_location="cpu", weights_only=False)
    chaos_cfg = ChaosConfig(**ck["cfg"])
    chaos_model = ChaosModel(chaos_cfg)
    chaos_model.load_state_dict(ck["model"])
    chaos_model.eval()
    with torch.no_grad():
        chaos_logits = chaos_model(idx)
    del sys.modules["chaos.config"], sys.modules["chaos.model"], sys.modules["chaos"]
    sys.path.remove(CHAOS_ROOT)

    sys.path.insert(0, VENTX_ROOT)
    from ventx.config import VentxConfig
    from ventx.model import VentxModel
    vck = torch.load(a.ventx_ckpt, map_location="cpu", weights_only=False)
    # Compare at Chaos-135M's native 2048 -- VenTX's own 65536/theta config is irrelevant
    # to whether the WEIGHTS were copied correctly; this test is about the import step.
    vcfg = VentxConfig(**{**vck["cfg"], "max_seq_len": a.seq_len, "rope_theta": chaos_cfg.rope_theta})
    ventx_model = VentxModel(vcfg)
    ventx_model.load_state_dict(vck["model"])
    ventx_model.eval()
    with torch.no_grad():
        ventx_logits = ventx_model(idx)

    diff = (chaos_logits - ventx_logits).abs()
    print(f"chaos logits:  shape={tuple(chaos_logits.shape)}, mean={chaos_logits.mean():.6f}, std={chaos_logits.std():.6f}")
    print(f"ventx logits:  shape={tuple(ventx_logits.shape)}, mean={ventx_logits.mean():.6f}, std={ventx_logits.std():.6f}")
    print(f"max abs diff:  {diff.max().item():.8f}")
    print(f"mean abs diff: {diff.mean().item():.8f}")

    argmax_match = (chaos_logits.argmax(-1) == ventx_logits.argmax(-1)).float().mean().item()
    print(f"argmax agreement across all positions: {argmax_match*100:.2f}%")

    if diff.max().item() < 1e-3 and argmax_match > 0.999:
        print("\nPASS: logits match within floating-point tolerance.")
    else:
        print("\nFAIL: logits diverge beyond floating-point tolerance -- import is NOT correct.")


if __name__ == "__main__":
    main()
