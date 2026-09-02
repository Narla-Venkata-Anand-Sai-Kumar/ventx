"""KV-cache incremental decode for VenTX-100K.

Kept separate from model.py so the training forward pass stays free of cache branching.
Dense causal attention means decode has no windowing/eviction logic to get right --
every generated token attends to the full accumulated cache, correct by construction,
unlike a sliding-window design that would need separate prefill-vs-steady-state cache
management. The real cost of that simplicity: decode gets slower as the cache grows,
since every step's attention is against the whole history so far -- expected, not a bug.
"""
import torch
import torch.nn.functional as F

from .model import apply_rope


class KVCache:
    """Per-layer K/V ring, preallocated to max_len."""

    def __init__(self, n_layer, batch, n_kv_head, head_dim, max_len, device, dtype=torch.bfloat16):
        self.k = [torch.zeros(batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(n_layer)]
        self.v = [torch.zeros(batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(n_layer)]

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


@torch.no_grad()
def _attend(attn, x, cos, sin, cache, layer, pos):
    B, T, _ = x.shape
    q = attn.q_proj(x).view(B, T, attn.nh, attn.hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, attn.nkv, attn.hd).transpose(1, 2)
    v = attn.v_proj(x).view(B, T, attn.nkv, attn.hd).transpose(1, 2)
    if attn.cfg.qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
    k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])

    cache.k[layer][:, :, pos:pos + T] = k
    cache.v[layer][:, :, pos:pos + T] = v
    k_all = cache.k[layer][:, :, :pos + T]
    v_all = cache.v[layer][:, :, :pos + T]

    if attn.rep > 1:
        k_all = k_all.repeat_interleave(attn.rep, dim=1)
        v_all = v_all.repeat_interleave(attn.rep, dim=1)

    # Prefill needs the causal mask; single-token decode attends to everything cached.
    o = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=(T > 1))
    o = o.transpose(1, 2).contiguous().view(B, T, attn.nh * attn.hd)
    return attn.o_proj(o)


@torch.no_grad()
def generate(model, idx, max_new_tokens=128, temperature=0.8, top_k=50, top_p=None,
            eos_id=None, logit_processor=None):
    """Sample continuations. `logit_processor(step, ids, logits) -> logits` is the hook
    grammar-constrained decoding plugs into."""
    model.eval()
    cfg = model.cfg
    B, T0 = idx.shape
    dev = idx.device
    total = T0 + max_new_tokens
    if total > cfg.max_seq_len:
        raise ValueError(f"{total} tokens exceeds max_seq_len {cfg.max_seq_len}")

    cache = KVCache(cfg.n_layer, B, cfg.n_kv_head, cfg.head_dim, total, dev)
    cos, sin = model.rope_cos.to(dev), model.rope_sin.to(dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    cur, pos = idx, 0
    for step in range(max_new_tokens + 1):
        x = model.embed(cur)
        for i, blk in enumerate(model.blocks):
            x = x + _attend(blk.attn, blk.n1(x), cos, sin, cache, i, pos)
            x = x + blk.ffn(blk.n2(x))
        logits = model.lm_head(model.norm_f(x[:, -1:]))[:, -1].float()
        pos += cur.shape[1]
        if step == max_new_tokens:
            break

        if logit_processor is not None:
            logits = logit_processor(step, idx, logits)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if top_p:
                sp, si = probs.sort(-1, descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p
                sp = torch.where(keep, sp, torch.zeros_like(sp))
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = si.gather(-1, torch.multinomial(sp, 1))
            else:
                nxt = torch.multinomial(probs, 1)

        if eos_id is not None:
            nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
            done |= nxt.squeeze(1) == eos_id
        idx = torch.cat([idx, nxt], dim=1)
        cur = nxt
        if eos_id is not None and bool(done.all()):
            break
    return idx, cache
