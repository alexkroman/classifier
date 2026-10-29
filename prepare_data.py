"""Build router train/test sets from the Loquacious Set (speechbrain/LoquaciousSet, `small` config).

Loquacious mixes read speech, spontaneous speech, parliament recordings, accented speakers and noisy
YouTube audio, so the two STT models naturally win on different clips. We only download a few shards
of `small/train` (two per source corpus) instead of the full 23 GB, plus the official test split.
"""

import argparse
from collections import Counter

from datasets import Audio, load_dataset

REPO = "speechbrain/LoquaciousSet"
# small/train shards are sorted by source corpus; take two shards per source:
# 5-6=voxpopuli, 15-16=librilight, 30-31=commonvoice, 45-46=peoples_speech, 60-61=yodas
TRAIN_SHARDS = [f"small/train-{i:05d}-of-00072.parquet" for i in (5, 6, 15, 16, 30, 31, 45, 46, 60, 61)]
TEST_SHARDS = "test/test-*.parquet"
SR = 16_000


def source_of(id_):
    """Source corpus from the clip ID format -- used for analysis only, never shown to the router."""
    if id_.startswith("common_voice"):
        return "commonvoice"
    if "-PLENARY-" in id_:
        return "voxpopuli"
    if "librivox" in id_:
        return "librilight"
    if id_.endswith(".wav"):
        return "yodas"
    return "peoples_speech"


def build_split(data_files, n, seed):
    ds = load_dataset(REPO, data_files=data_files, split="train")
    # Whisper sees at most 30s per window; keep clips both models can handle in one pass.
    ds = ds.filter(lambda d, t: 1.0 <= d <= 30.0 and bool(t.strip()), input_columns=["duration", "text"])
    ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda i: {"source": source_of(i)}, input_columns=["ID"])
    return ds.rename_column("wav", "audio").cast_column("audio", Audio(sampling_rate=SR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=0, help="0 = full test split")
    args = ap.parse_args()

    for name, files, n, seed in [("train", TRAIN_SHARDS, args.n_train, 0), ("test", TEST_SHARDS, args.n_test, 1)]:
        ds = build_split(files, n, seed)
        ds.save_to_disk(f"data/{name}")
        print(name, len(ds), "clips,", round(sum(ds["duration"]) / 3600, 2), "hours")
        print(Counter(ds["source"]))
