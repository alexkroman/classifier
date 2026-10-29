"""Build a domain-balanced router dataset from Open ASR Leaderboard (ESB) test sets, scored by every model.

Why: on Loquacious, one model (the NeMo Conformer) was trained on most of the source corpora and simply wins
everywhere, so there is little for a router to learn. These six domains are ones where the models split:
far-field meetings, accented earnings calls, finance calls, podcasts/YouTube, audiobooks, parliament.

Every domain contributes the same number of clips, and each domain is split 70/15/15 into train/val/test
*separately* by speaker or recording, so all three splits have the identical domain mix and never share a voice.
Common Voice is left out: its IDs are anonymous hashes, so speakers couldn't be kept on one side of a split.

Output: data/mix/{train,val,test}_scored (same schema as data/test_scored, plus `group`).
"""

import argparse
from collections import Counter

import numpy as np
from datasets import Audio, concatenate_datasets, load_dataset

from transcribe import normalize, score

REPO = "hf-audio/open-asr-leaderboard"


def spaced(prefix, n_shards, take):
    """Shards are sorted by clip duration, so take evenly spaced ones to cover every length range."""
    idx = np.unique(np.linspace(0, n_shards - 1, take).round().astype(int))
    return [f"{prefix}-{i:05d}-of-{n_shards:05d}.parquet" for i in idx]


# domain -> (shard files, ID -> speaker/recording group)
DOMAINS = {
    # AMI meeting series reuse the same 4 people across sessions, so group by speaker, not meeting
    "ami": (spaced("ami/test", 15, 6), lambda i: i.split("_")[3]),
    "earnings22": (spaced("earnings22/test", 5, 5), lambda i: i.split("/")[0]),  # earnings call
    "spgispeech": (spaced("spgispeech/test", 38, 4), lambda i: i.split("/")[0]),  # earnings call
    "gigaspeech": (spaced("gigaspeech/test", 19, 4), lambda i: i.split("_")[0]),  # podcast / video file
    "librispeech": (
        ["librispeech/test.clean-00000-of-00001.parquet", "librispeech/test.other-00000-of-00001.parquet"],
        lambda i: i.split("-")[0],  # reader
    ),
    "voxpopuli": (spaced("voxpopuli/test", 4, 4), lambda i: i.split("-PLENARY")[0]),  # parliament session
}
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_splits(groups, rng):
    """Largest groups first, each into the split furthest below its clip target (ties broken randomly).

    Filling splits with groups in random order overshoots badly when a domain has only a few big groups
    (Earnings-22 has just 6 calls, AMI 16 speakers); largest-first keeps every domain close to 70/15/15.
    """
    counts = Counter(groups)
    order = sorted(counts, key=lambda g: (-counts[g], rng.random()))
    split_of, filled, total = {}, Counter(), len(groups)
    for g in order:
        split_of[g] = max(FRACS, key=lambda s: FRACS[s] * total - filled[s])
        filled[split_of[g]] += counts[g]
    return [split_of[g] for g in groups]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=2000)
    cli = ap.parse_args()
    rng = np.random.default_rng(0)

    parts = []
    for name, (files, group_of) in DOMAINS.items():
        ds = load_dataset(REPO, data_files=files, split="train")
        # same 1-30s window as the Loquacious set: every model handles the clip in a single pass
        ds = ds.filter(
            lambda d, t: 1.0 <= d <= 30.0 and bool(normalize(t).strip()), input_columns=["audio_length_s", "text"]
        )
        ds = ds.shuffle(seed=0).select(range(min(cli.per_domain, len(ds))))
        groups = [group_of(i) for i in ds["id"]]
        ds = ds.select_columns(["audio", "text", "id", "audio_length_s"])
        ds = ds.add_column("source", [name] * len(ds)).add_column("group", groups)
        ds = ds.add_column("split", assign_splits(groups, rng))
        parts.append(ds)
        c = Counter(ds["split"])
        print(
            f"{name:12s} {len(ds):5d} clips, {len(set(groups)):4d} groups  "
            + "  ".join(f"{s} {c[s] / len(ds):.0%}" for s in ("train", "val", "test"))
        )

    ds = concatenate_datasets(parts).rename_columns({"id": "ID", "audio_length_s": "duration"})
    ds = score(ds.cast_column("audio", Audio(sampling_rate=16_000)), desc="mix")
    for split in ("train", "val", "test"):
        part = ds.filter(lambda s, split=split: s == split, input_columns=["split"])
        part.save_to_disk(f"data/mix/{split}_scored")
        print(split, len(part), "clips", dict(sorted(Counter(part["source"]).items())))
