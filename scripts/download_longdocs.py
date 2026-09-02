"""Pull the raw long-document corpora for context-extension training.

Real, verified sources (checked live via the HF Hub API before writing this script --
the original bookcorpus/bookcorpusopen repo is gated/removed, moved under the
defunct-datasets namespace, so this uses the community re-upload that preserves the
same 17,868 whole-book documents):

  - lucadiliello/bookcorpusopen  -- 14 parquet shards, `text`+`title` columns,
    17,868 full books, ~6.6GB uncompressed text. Whole-book (not sentence-shuffled)
    text is exactly what long-context training needs.
  - sedthh/gutenberg_english     -- 37 parquet shards, `TEXT`+`SOURCE`+`METADATA`
    columns, 48,284 full Project Gutenberg books, ~18GB uncompressed text.

Pulls specific shards, not the whole repo, same discipline as Chaos-135M's
download_data.py -- a couple of shards from each source already yields far more
tokens than the ~140M-token long-doc budget (70% of the ~200M-token context-extension
budget) needs.
"""
import argparse
import os
import time

from huggingface_hub import hf_hub_download

RAW = os.environ.get("VENTX_RAW", "C:/ventx-data/raw")

BOOKCORPUS_FILES = [
    "data/train-00000-of-00014-e40347a4a9a752dd.parquet",
    "data/train-00001-of-00014-4f769efe80e66fc3.parquet",
]
GUTENBERG_FILES = [
    "data/train-00000-of-00037-f5fce855b93d2d02.parquet",
    "data/train-00001-of-00037-9f227d74fc154ce9.parquet",
]


def fetch(repo, fname, subdir):
    dest = os.path.join(RAW, subdir)
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, os.path.basename(fname))
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target, 0.0, os.path.getsize(target)
    t0 = time.time()
    p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                        local_dir=dest, cache_dir=os.path.join(RAW, ".hfcache"))
    if os.path.abspath(p) != os.path.abspath(target):
        os.replace(p, target)
    return target, time.time() - t0, os.path.getsize(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bookcorpus", type=int, default=2, help="bookcorpusopen shards (of 14)")
    ap.add_argument("--gutenberg", type=int, default=2, help="gutenberg shards (of 37)")
    a = ap.parse_args()

    jobs = [("lucadiliello/bookcorpusopen", f, "bookcorpusopen") for f in BOOKCORPUS_FILES[:a.bookcorpus]]
    jobs += [("sedthh/gutenberg_english", f, "gutenberg") for f in GUTENBERG_FILES[:a.gutenberg]]

    total = 0
    for i, (repo, fname, sub) in enumerate(jobs, 1):
        try:
            path, dt, size = fetch(repo, fname, sub)
            total += size
            rate = (size / 1e6 / dt) if dt > 0 else float("inf")
            tag = "cached" if dt == 0 else f"{dt:6.1f}s  {rate:6.1f} MB/s"
            print(f"[{i:2d}/{len(jobs)}] {sub:14s} {size/1e6:8.1f} MB  {tag}", flush=True)
        except Exception as e:
            print(f"[{i:2d}/{len(jobs)}] FAIL {repo}/{fname}: {type(e).__name__}: {e}", flush=True)
    print(f"\nTOTAL raw: {total/1e9:.2f} GB in {RAW}")


if __name__ == "__main__":
    main()
