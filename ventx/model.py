"""VenTX-100K: decoder-only transformer, native long-context (dense causal attention,
no sliding window / no interleaved local-global compromise -- every layer attends to the
full sequence, by deliberate design choice for this project).

Parameter names/shapes are pinned to match Chaos-135M's trained checkpoint (see
ventx/config.py's docstring) so that checkpoint's weights load directly via a plain
state_dict copy. The code itself is a fresh implementation, not copied.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VentxConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd) -- rotate-half formulation. cos/sin built in fp32, rotation runs
    in the activation dtype (upcasting x to fp32 here costs real memory for no benefit)."""
    T = x.shape[-2]
    cos = cos[:T].view(1, 1, T, -1).to(x.dtype)
    sin = sin[:T].view(1, 1, T, -1).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """GQA with QK-norm, dense causal attention (native long context -- no window)."""

    def __init__(self, cfg: VentxConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        assert self.nh % self.nkv == 0, "n_head must be divisible by n_kv_head"
        self.rep = self.nh // self.nkv

        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd, cfg.rms_eps)
            self.k_norm = RMSNorm(self.hd, cfg.rms_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Explicit repeat, not F.scaled_dot_product_attention(enable_gqa=True): that flag
        # has no fused kernel on this stack and silently drops to the unfused MATH backend,
        # materializing the full B x H x T x T score matrix -- a real, previously-found
        # blowup (6.68GB vs 0.29GB measured at B8/T2048 in the reference project). The
        # explicit repeat keeps the flash-attention kernel path.
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(o)


class SwiGLU(nn.Module):
    """Chunks the sequence dimension internally when it's long enough that the full
    (B,T,ffn_hidden) intermediate becomes the dominant activation. ffn_hidden=2816 is a
    3.67x expansion over d_model=768 -- at T=100,000 that intermediate alone is larger
    than the attention op's own memory footprint (measured separately at ~0.72GB for the
    whole attention op at this length). Per MegaTrain (single-GPU 100B+-param training up
    to 512K context): "chunked MLP execution keeps memory within bounds without degrading
    throughput" -- same idea already used for the loss head, applied here too.
    """
    CHUNK = 2048

    def __init__(self, cfg: VentxConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        # Instance attribute shadows the class default so a large-VRAM GPU can raise it
        # past any real seq_len (see VentxModel.set_chunk_sizes) and skip chunking --
        # and the checkpoint() calls that come with it -- entirely. checkpoint() is
        # documented as incompatible with CUDA graph capture (banned RNG state ops
        # during stream capture), so on a GPU with room to spare this chunking, built
        # for an 8GB budget, is pure kernel-launch overhead with no memory benefit.
        self.CHUNK = self.CHUNK

    def _piece(self, xc):
        return self.down(F.silu(self.gate(xc)) * self.up(xc))

    def forward(self, x):
        B, T, D = x.shape
        if T <= self.CHUNK:
            return self._piece(x)
        out = torch.empty_like(x)
        for i in range(0, T, self.CHUNK):
            xc = x[:, i:i + self.CHUNK]
            if self.training and torch.is_grad_enabled():
                out[:, i:i + self.CHUNK] = torch.utils.checkpoint.checkpoint(
                    self._piece, xc, use_reentrant=False)
            else:
                out[:, i:i + self.CHUNK] = self._piece(xc)
        return out


class Block(nn.Module):
    def __init__(self, cfg: VentxConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg, layer_idx)
        self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


class VentxModel(nn.Module):
    def __init__(self, cfg: VentxConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._grad_checkpoint = False
        self._ckpt_group = 1   # blocks per checkpoint boundary; >1 trades recompute cost
                                # for fewer boundary tensors held alive across the forward
                                # pass (each held tensor is small alone, but accumulates
                                # linearly with n_layer at long T -- see plan notes)
        self._loss_chunk = 2048
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_rope_theta(self, theta: float, max_seq_len: Optional[int] = None):
        """Rebuild the RoPE cache at a new theta/length without reconstructing the model --
        weights are untouched, only the (non-persistent, never-checkpointed) cos/sin
        buffers change."""
        if max_seq_len:
            self.cfg.max_seq_len = max_seq_len
        dev = self.rope_cos.device
        cos, sin = build_rope_cache(self.cfg.head_dim, self.cfg.max_seq_len, theta, dev)
        self.rope_cos, self.rope_sin = cos, sin

    def enable_grad_checkpoint(self, flag: bool = True, group: int = 1):
        self._grad_checkpoint = flag
        self._ckpt_group = max(1, group)

    def set_chunk_sizes(self, ffn_chunk: Optional[int] = None, loss_chunk: Optional[int] = None):
        """Raise (or restore) the FFN/loss chunk sizes. Chunking exists to bound memory
        on small GPUs; each chunk boundary is a torch.utils.checkpoint.checkpoint() call,
        which is real per-step CPU-launch overhead and is documented as incompatible with
        CUDA graph capture. On a GPU with room to spare, setting a chunk size past the
        actual seq_len makes that module take its un-chunked, checkpoint-free fast path."""
        if ffn_chunk is not None:
            for blk in self.blocks:
                blk.ffn.CHUNK = ffn_chunk
        if loss_chunk is not None:
            self._loss_chunk = loss_chunk

    @staticmethod
    def _run_group(blocks, x, cos, sin):
        for blk in blocks:
            x = blk(x, cos, sin)
        return x

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T].to(x.device), self.rope_sin[:T].to(x.device)

        if self._grad_checkpoint and self.training:
            g = self._ckpt_group
            for i in range(0, len(self.blocks), g):
                grp = self.blocks[i:i + g]
                x = torch.utils.checkpoint.checkpoint(
                    self._run_group, grp, x, cos, sin, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x, cos, sin)

        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x)
        return None, self._chunked_loss(x, targets, chunk=self._loss_chunk)

    def _chunked_loss(self, x, targets, chunk: int = 2048):
        """Cross-entropy over row-chunks, recomputing each chunk's logits in backward.

        At d_model=768/vocab=49152, a full fp32 logits tensor at long T dwarfs every other
        activation (~19.6GB at T=100,000, B=1) -- chunking with checkpointing caps the live
        logits at chunk x vocab regardless of sequence length.
        """
        xf = x.reshape(-1, x.size(-1))
        tf = targets.reshape(-1)
        n_valid = (tf != -100).sum().clamp(min=1)

        def piece(xc, tc):
            logits = F.linear(xc, self.lm_head.weight)
            return F.cross_entropy(logits.float(), tc, ignore_index=-100, reduction="sum")

        if xf.size(0) <= chunk:
            return piece(xf, tf) / n_valid

        total = None
        for i in range(0, xf.size(0), chunk):
            xc, tc = xf[i:i + chunk], tf[i:i + chunk]
            if self.training and torch.is_grad_enabled():
                part = torch.utils.checkpoint.checkpoint(piece, xc, tc, use_reentrant=False)
            else:
                part = piece(xc, tc)
            total = part if total is None else total + part
        return total / n_valid

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self, seq_len: Optional[int] = None) -> float:
        """Forward+backward FLOPs/token: 6N for the matmuls plus the dense attention term.

        No window discount here -- every layer is full causal attention at whatever
        seq_len is passed, by this project's design (native long context, not windowed).
        """
        c = self.cfg
        T = seq_len or c.max_seq_len
        n = self.num_params(non_embedding=True) + c.vocab_size * c.d_model
        attn = 12 * c.n_layer * c.d_model * T   # 6 * 2 * L * d * T, QK^T and AV, fwd+bwd
        return 6 * n + attn
