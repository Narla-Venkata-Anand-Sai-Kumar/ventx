"""Muon optimizer: Newton-Schulz orthogonalized momentum for 2D hidden-layer matrices,
AdamW for everything else (embeddings, norms, and the tied lm_head). Fresh implementation
of the published algorithm (Keller Jordan's Muon) -- not copied from the reference project.
"""
import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonalization of G via a quintic Newton-Schulz iteration,
    computed in bfloat16 for speed. Assumes len(G.shape) == 2."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)   # Nesterov-style lookahead
                g = zeropower_via_newtonschulz5(g)
                p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def split_params(model: nn.Module):
    """2D hidden-layer weight matrices -> Muon; everything else (embeddings, norms, the
    tied lm_head) -> AdamW. Muon is only well-behaved on matrices with two large,
    comparable dimensions -- embedding tables and 1D norm weights don't fit that shape."""
    muon_params, adamw_params = [], []
    embed_ptr = model.embed.weight.data_ptr() if hasattr(model, "embed") else None
    for name, p in model.named_parameters():
        if p.data_ptr() == embed_ptr or p.ndim < 2 or "norm" in name:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params


def build_optimizer(model: nn.Module, tcfg, adamw_8bit: bool = False):
    """adamw_8bit=True swaps AdamW for bitsandbytes' PagedAdamW8bit (embeddings/norms only
    -- Muon's own momentum buffer isn't touched by this flag). Real memory saving at long
    context: the tied embed table is this model's single largest AdamW-managed tensor
    (~37.75M params), and paging lets its state spill to CPU RAM on demand rather than
    permanently reserving GPU memory for it."""
    muon_params, adamw_params = split_params(model)
    muon = Muon(muon_params, lr=tcfg.lr_matrix, momentum=tcfg.muon_momentum,
               weight_decay=tcfg.weight_decay)
    if adamw_8bit:
        import bitsandbytes as bnb
        adamw = bnb.optim.PagedAdamW8bit(adamw_params, lr=tcfg.lr_embed, betas=tcfg.adam_betas,
                                         weight_decay=tcfg.weight_decay)
    else:
        # fused=True runs the AdamW update as a single fused CUDA kernel instead of the
        # default per-parameter Python loop -- a real, free kernel-launch reduction on
        # the optimizer step specifically, separate from anything already tried on the
        # model's own forward/backward.
        adamw = torch.optim.AdamW(adamw_params, lr=tcfg.lr_embed, betas=tcfg.adam_betas,
                                  weight_decay=tcfg.weight_decay, fused=True)

    class Combined:
        """Presents both optimizers as one, so train.py's single opt.step()/param_groups
        loop doesn't need to know two optimizer families are involved."""
        def __init__(self, opts):
            self.opts = opts
            self.param_groups = [g for o in opts for g in o.param_groups]

        def step(self):
            for o in self.opts:
                o.step()

        def zero_grad(self, set_to_none=True):
            for o in self.opts:
                o.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return [o.state_dict() for o in self.opts]

        def load_state_dict(self, states):
            for o, s in zip(self.opts, states):
                o.load_state_dict(s)

    return Combined([muon, adamw])
