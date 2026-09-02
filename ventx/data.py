"""Tokenized-shard IO for VenTX-100K.

Same on-disk design as Chaos-135M's shard format (flat uint16 tokens behind a small
header) -- fresh implementation, not copied, but there is no reason to invent a new
format: uint16 is correct as long as vocab_size < 65536 (49152 here, same tokenizer),
and a magic+length header catches truncated/corrupt shards instead of silently training
on garbage. Long documents carry no boundary metadata beyond the <|endoftext|> token
already baked into the stream by the tokenizer step, exactly as in the reference project.
"""
import json
import os

import numpy as np
import torch

MAGIC = 0x56545830          # "VTX0"
HEADER_BYTES = 256
HEADER_INTS = HEADER_BYTES // 4


def write_shard(path, tokens: np.ndarray):
    assert tokens.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def read_shard(path) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32)
        if header[0] != MAGIC:
            raise ValueError(f"{path}: bad magic (not a VenTX token shard)")
        n = int(header[2])
        toks = np.frombuffer(f.read(), dtype=np.uint16)
    if len(toks) != n:
        raise ValueError(f"{path}: truncated -- header says {n}, found {len(toks)}")
    return toks


def shard_paths(data_dir, split):
    idx = os.path.join(data_dir, f"{split}_index.json")
    with open(idx) as f:
        meta = json.load(f)
    return [os.path.join(data_dir, s["file"]) for s in meta["shards"]], meta


class ShardDataLoader:
    """Streams contiguous (B, T+1) windows from memory-mapped shards, sequential within
    a shard, shards shuffled per epoch. At VenTX's context lengths a single (B=1, T=65536)
    window is already 65,537 tokens -- shards must stay well above that or every batch
    forces a shard advance."""

    def __init__(self, data_dir, split, batch_size, seq_len, seed=1337, device="cuda"):
        self.paths, self.meta = shard_paths(data_dir, split)
        if not self.paths:
            raise ValueError(f"no shards for split {split} in {data_dir}")
        self.B, self.T = batch_size, seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.total_tokens = int(self.meta["total_tokens"])
        self.epoch = 0
        self._new_epoch()

    def _new_epoch(self):
        self.order = self.rng.permutation(len(self.paths))
        self.shard_i = 0
        self._load_shard()

    def _load_shard(self):
        path = self.paths[self.order[self.shard_i]]
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
        self.pos = 0

    def _advance_shard(self):
        self.shard_i += 1
        if self.shard_i >= len(self.paths):
            self.epoch += 1
            self._new_epoch()
        else:
            self._load_shard()

    def next_batch(self):
        need = self.B * self.T + 1
        if self.pos + need > len(self.tokens):
            self._advance_shard()
        buf = np.asarray(self.tokens[self.pos:self.pos + need], dtype=np.int64)
        self.pos += self.B * self.T
        x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        y = torch.from_numpy(buf[1:]).view(self.B, self.T)
        if self.device == "cuda":
            x = x.pin_memory().to("cuda", non_blocking=True)
            y = y.pin_memory().to("cuda", non_blocking=True)
        return x, y
