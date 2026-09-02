"""Tokenize the long-document corpora (BookCorpusOpen + Gutenberg) into VenTX .bin
shards, mixed with a slab of Chaos-135M's already-tokenized short-doc corpus
(fineweb-edu/cosmopedia/finemath/sql-synth) for forgetting mitigation.

Reuses the exact same BPE tokenizer Chaos-135M's checkpoint was trained with
(C:/chaos-data/tokenizer/chaos-bpe.json) -- this is not optional: the imported
embedding table only means anything under the vocab it was trained on.

Deliberate deviation from Chaos-135M's tokenize_corpus.py reference pattern: documents
here are whole books (tens of thousands of words), not web pages, so batches handed to
encoder workers are small (default 8, not 1000) -- the reference's batch size would hold
tens of thousands of full books in flight at once against 3-4GB typically-free RAM.

Mixing strategy: long documents are queued whole and separated by <|endoftext|>, so any
window sampled during training that lands inside one sees a genuinely continuous book,
not a patchwork -- the whole point of this corpus. Short-doc content is spliced in as
large, coarse slabs (millions of tokens at a time) read directly from the existing
tokenized short-doc shards, not finely interleaved, for the same reason: fine-grained
mixing would chop books into pieces smaller than the context lengths being trained for.
"""
import argparse
import glob
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ventx.data import write_shard   # noqa: E402

RAW = os.environ.get("VENTX_RAW", "C:/ventx-data/raw")
TOKDIR = os.environ.get("CHAOS_TOK", "C:/chaos-data/tokenizer")
OUT = os.environ.get("VENTX_TOK", "C:/ventx-data/tokenized")
SHORT_DOC_DIR = os.environ.get("CHAOS_TOK_BIN", "C:/chaos-data/tokenized")

# Chaos-135M's own shard format (CHA0 magic) -- read directly, not via ventx.data
# (that module only accepts its own VTX0 magic, on purpose: the two projects' shards
# should never be silently interchangeable elsewhere in the pipeline).
_CHA0_MAGIC = 0x43484130
_CHA0_HEADER_BYTES = 256

_tok = None


def _init():
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(os.path.join(TOKDIR, "chaos-bpe.json"))


def _encode(docs):
    eot = _tok.token_to_id("<|endoftext|>")
    out = []
    for enc in _tok.encode_batch(docs):
        if len(enc.ids) < 256:      # skip fragments too short to be useful long-context signal
            continue
        out.extend(enc.ids)
        out.append(eot)
    return np.array(out, dtype=np.uint16)


def docs_column(path, column, batch_size):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column]):
        vals = [v.as_py() for v in batch.column(0) if v.as_py()]
        for i in range(0, len(vals), batch_size):
            yield vals[i:i + batch_size]


SOURCES = [
    ("bookcorpusopen", "text"),
    ("gutenberg", "TEXT"),
]


def long_doc_batches(batch_size):
    gens = []
    for sub, col in SOURCES:
        files = sorted(glob.glob(os.path.join(RAW, sub, "*.parquet")))
        if not files:
            print(f"  (skipping {sub}: no files -- run scripts/download_longdocs.py first)")
            continue
        print(f"  source: {sub} ({len(files)} shard file(s))", flush=True)
        for path in files:
            gens.append(docs_column(path, col, batch_size))
    # Round-robin across sources so long shards aren't all bookcorpus-then-all-gutenberg.
    active = deque(gens)
    while active:
        gen = active.popleft()
        try:
            yield next(gen)
            active.append(gen)
        except StopIteration:
            pass


class ShortDocReader:
    """Sequential cursor over Chaos-135M's existing tokenized short-doc train shards,
    for pulling large contiguous slabs to mix into the long-doc stream."""

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "train_index.json")) as f:
            meta = json.load(f)
        self.paths = [os.path.join(data_dir, s["file"]) for s in meta["shards"]]
        if not self.paths:
            raise ValueError(f"no short-doc shards found in {data_dir}")
        self.shard_i = 0
        self.pos = 0
        self._load()

    def _load(self):
        with open(self.paths[self.shard_i], "rb") as f:
            header = np.frombuffer(f.read(_CHA0_HEADER_BYTES), dtype=np.int32)
            assert header[0] == _CHA0_MAGIC, f"{self.paths[self.shard_i]}: bad magic"
        self.tokens = np.memmap(self.paths[self.shard_i], dtype=np.uint16, mode="r",
                                offset=_CHA0_HEADER_BYTES)
        self.pos = 0

    def take(self, n):
        out = []
        remaining = n
        while remaining > 0:
            avail = len(self.tokens) - self.pos
            if avail <= 0:
                self.shard_i = (self.shard_i + 1) % len(self.paths)
                self._load()
                continue
            grab = min(avail, remaining)
            out.append(np.asarray(self.tokens[self.pos:self.pos + grab]))
            self.pos += grab
            remaining -= grab
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-tokens", type=int, default=230_000_000,
                    help="target long-document token budget (~70%% of the mix)")
    ap.add_argument("--short-frac", type=float, default=0.30,
                    help="fraction of final corpus that is short-doc mix-in")
    ap.add_argument("--short-chunk-tokens", type=int, default=2_000_000,
                    help="size of each spliced-in short-doc slab")
    ap.add_argument("--shard-tokens", type=int, default=50_000_000)
    ap.add_argument("--val-docs", type=int, default=8,
                    help="whole long documents held out for validation (unmixed)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    tok = __import__("tokenizers").Tokenizer.from_file(os.path.join(TOKDIR, "chaos-bpe.json"))
    eot = tok.token_to_id("<|endoftext|>")
    short_reader = ShortDocReader(SHORT_DOC_DIR)

    t0 = time.time()
    shards = []
    buf, buf_n = [], 0
    long_written = 0
    short_written = 0
    docs_seen = 0
    val_docs_written = 0
    split = "val"

    def flush(split_name):
        nonlocal buf, buf_n
        if buf_n == 0:
            return
        arr = np.concatenate(buf)
        name = f"{split_name}_{len([s for s in shards if s['split']==split_name]):04d}.bin"
        write_shard(os.path.join(OUT, name), arr)
        shards.append(dict(file=name, split=split_name, tokens=int(len(arr))))
        print(f"  wrote {name}  {len(arr)/1e6:7.1f}M tokens  "
              f"(long={long_written/1e6:.1f}M short={short_written/1e6:.1f}M, "
              f"{time.time()-t0:.0f}s)", flush=True)
        buf, buf_n = [], 0

    window = a.workers * 4
    src = long_doc_batches(a.batch_size)
    # spawn, not the platform default fork -- see tokenize_corpus.py's identical fix:
    # PyArrow's parent-process IO thread pool does not survive fork() cleanly on Linux.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, mp_context=ctx) as ex:
        pending = deque()
        for _ in range(window):
            try:
                pending.append(ex.submit(_encode, next(src)))
            except StopIteration:
                break
        while pending:
            arr = pending.popleft().result()
            try:
                pending.append(ex.submit(_encode, next(src)))
            except StopIteration:
                pass
            if len(arr) == 0:
                continue

            if split == "val":
                buf.append(arr); buf_n += len(arr); long_written += len(arr)
                val_docs_written += 1
                if val_docs_written >= a.val_docs:
                    flush("val")
                    split = "train"
                continue

            buf.append(arr)
            buf_n += len(arr)
            long_written += len(arr)
            docs_seen += 1

            # Coarse-grained short-doc splice: whenever the running short fraction
            # falls behind target, inject one big slab (not fine interleaving).
            total_so_far = long_written + short_written
            if total_so_far > 0 and short_written / total_so_far < a.short_frac:
                slab = short_reader.take(a.short_chunk_tokens)
                buf.append(slab)
                buf.append(np.array([eot], dtype=np.uint16))
                buf_n += len(slab) + 1
                short_written += len(slab) + 1

            if buf_n >= a.shard_tokens:
                flush("train")
            if long_written >= a.long_tokens:
                break
    flush(split if split == "train" else "train")

    for sp in ("train", "val"):
        sl = [s for s in shards if s["split"] == sp]
        with open(os.path.join(OUT, f"{sp}_index.json"), "w") as f:
            json.dump(dict(shards=sl, total_tokens=sum(s["tokens"] for s in sl)), f, indent=2)

    tr = sum(s["tokens"] for s in shards if s["split"] == "train")
    va = sum(s["tokens"] for s in shards if s["split"] == "val")
    print(f"\ntrain {tr/1e6:.1f}M tokens | val {va/1e6:.1f}M | long={long_written/1e6:.1f}M "
          f"short={short_written/1e6:.1f}M | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
