"""VenTX-100K model + training configuration.

225M-parameter architecture (226.1M measured via VentxModel.num_params()), trained fully
from scratch -- NOT bootstrapped from Chaos-135M's 134.5M checkpoint. That checkpoint's
weight-import path (scripts/import_chaos_weights.py, scripts/verify_import.py) is now
historical: a real, verified-working technique for the 134.5M size, kept in the repo as
a record, but no longer part of this project's active pipeline now that the param count
changed and the shapes no longer match.

The 225M dims are a uniform scale-up of Chaos-135M's proven shape, not a new design:
same head_dim (64), same GQA ratio (n_head:n_kv_head = 3:1), same tied-embedding /
qk-norm / SwiGLU choices -- only d_model (768->960), n_layer (12->14), and ffn_hidden
(2816->3584) grew. max_seq_len/rope_theta below are this project's context-extension
target; the base pretrain runs at a short context (see ventx/train.py) before staging
up to these.
"""
from dataclasses import dataclass


@dataclass
class VentxConfig:
    # --- architecture: 225M scale-up of Chaos-135M's shape, trained from scratch ---
    vocab_size: int = 49152
    n_layer: int = 14
    d_model: int = 960
    n_head: int = 15          # query heads
    n_kv_head: int = 5        # GQA key/value heads (3:1 ratio, same as Chaos-135M)
    ffn_hidden: int = 3584
    qk_norm: bool = True
    tie_embeddings: bool = True
    rms_eps: float = 1e-6
    dropout: float = 0.0

    # --- positional: VenTX-100K's own, not inherited from Chaos-135M ---
    # 65,536 is the real, Stage-0-confirmed ceiling on this hardware (RTX 5060 8GB) --
    # 100K and even ~86K were tested and found unreliable (OOM / silent WDDM memory
    # spillover degrading to ~100x slower rather than failing cleanly). Locked in per
    # explicit direction after real measurement, not the original aspirational target.
    max_seq_len: int = 65_536
    rope_theta: float = 362_000.0   # NTK-aware scaled starting point for 65,536, see plan

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class TrainConfig:
    micro_batch: int = 1
    grad_accum: int = 16
    seq_len: int = 100_000

    total_steps: int = 20_000
    warmup_steps: int = 500
    decay_frac: float = 0.20

    lr_matrix: float = 0.02
    lr_embed: float = 0.002
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    adam_betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    grad_checkpoint: bool = True
    seed: int = 1337
