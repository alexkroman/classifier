"""Build the router dataset: speaker-diverse 10-29.5s chunks of one voice, scored by every candidate STT model.

Why chunks: on short clips the per-clip labels are mostly noise (many ties, or a one-word difference), while which
model wins *is* persistent within a speaker/recording, so longer stretches of one voice accumulate that signal.
Why many voices: the signal left beyond domain detection is per speaker/recording (a voice's two random halves agree
on whisper vs conformer at r = +0.55), so the hours are spread over as many voices as possible: every held-out split
we can get (ESB test sets plus the corpora's dev splits, none of which the candidate models train on), with each
recording capped at a few chunks.

Sources per domain: AMI test + AMI IHM dev; all 125 Earnings-22 calls; SPGISpeech test (all shards); GigaSpeech
test; LibriSpeech test + dev; VoxPopuli test + dev; EdAcc. Consecutive segments of one recording+speaker are packed
into chunks of a random 10-28s target (never over 29.5s, so Whisper sees the chunk in one window), joined with 0.2s
of silence, and every chunk is re-transcribed by every model: the labels are what each model does on that audio.
Splits are 70/15/15 per domain by speaker/recording, so all three have the same domain mix and never share a voice.

Parquet is read remotely with column projection: metadata first, then only the row groups holding audio we use,
so nothing is mirrored into the HF caches.

Output: data/speakers/{train,val,test}_scored.
"""

import argparse
import io
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset, Features, Value, concatenate_datasets, load_from_disk
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from transcribe import normalize, score

OUT = "data/speakers"
SR = 16_000
GAP = 0.2  # seconds of silence between joined segments
MAX_DUR = 29.5
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
LB = "datasets/hf-audio/open-asr-leaderboard"
fs = HfFileSystem()


@dataclass
class Source:
    """One parquet collection and how to read its columns into (id, text, start, duration)."""

    glob: str
    id_col: str
    text_col: str
    dur: Callable | None = None  # row dict -> seconds; None = measure from the audio
    make_id: Callable | None = None  # (row dict, file index, row index) -> id for ordering / keys (default: id_col)
    start: Callable | None = None  # row dict -> start time for ordering within a recording (default: natural id)


@dataclass
class Domain:
    sources: list
    chunk_key: Callable  # id -> one recording + speaker
    group_of: Callable  # id -> speaker / recording group for splits


lb_dur = lambda r: r["audio_length_s"]  # noqa: E731
DOMAINS = {
    "ami": Domain(
        [
            Source(f"{LB}/ami/test-*.parquet", "id", "text", lb_dur),
            Source(
                "datasets/edinburghcstr/ami/ihm/validation-*.parquet",
                "audio_id",
                "text",
                lambda r: r["end_time"] - r["begin_time"],
            ),
        ],
        lambda i: "_".join(i.split("_")[1:4]),  # meeting + headset + speaker
        lambda i: i.split("_")[3],  # speaker (reused across meetings)
    ),
    "earnings22": Domain(
        [
            Source(
                "datasets/distil-whisper/earnings22/chunked/test-*.parquet",
                "file_id",
                "transcription",
                lambda r: r["end_ts"] - r["start_ts"],
                make_id=lambda r, fi, k: f"{r['file_id']}/{int(r['start_ts'] * 100):08d}",  # sortable by time
            )
        ],
        lambda i: i.split("/")[0],  # call
        lambda i: i.split("/")[0],
    ),
    "spgispeech": Domain(
        [Source(f"{LB}/spgispeech/test-*.parquet", "id", "text", lb_dur)],
        lambda i: i.split("/")[0],
        lambda i: i.split("/")[0],
    ),
    "gigaspeech": Domain(
        [Source(f"{LB}/gigaspeech/test-*.parquet", "id", "text", lb_dur)],
        lambda i: i.split("_")[0],
        lambda i: i.split("_")[0],
    ),
    "librispeech": Domain(
        [
            Source(f"{LB}/librispeech/test.*.parquet", "id", "text", lb_dur),
            Source("datasets/openslr/librispeech_asr/all/validation.*/*.parquet", "id", "text"),
        ],
        lambda i: i.rsplit("-", 1)[0],  # reader + chapter
        lambda i: i.split("-")[0],  # reader
    ),
    # Edinburgh International Accents of English: conversations between pairs, 51 L1s, held-out ASR eval set.
    # No ids or times: rows are in conversation order, so order by (file, row). Split by conversation, not
    # speaker: the two sides share the recording setup and bleed into each other's microphones.
    "edacc": Domain(
        [
            Source(
                "datasets/edinburghcstr/edacc/data/*.parquet",
                "speaker",
                "text",
                make_id=lambda r, fi, k: f"{r['speaker']}/{fi:02d}{k:06d}",
            )
        ],
        lambda i: i.split("/")[0],  # one speaker of one conversation
        lambda i: i.split("/")[0].rsplit("-", 1)[0],  # conversation
    ),
    "voxpopuli": Domain(
        [
            Source(f"{LB}/voxpopuli/test-*.parquet", "id", "text", lb_dur),
            Source("datasets/facebook/voxpopuli/en/validation-*.parquet", "audio_id", "raw_text"),
        ],
        lambda i: i.rsplit("_", 1)[0],  # one speech
        lambda i: i.split("-PLENARY")[0],  # parliament session
    ),
}


def read_metadata(src):
    """[(file, row, id, text, duration)] for every row of a source, reading only the small columns."""
    rows = []
    for fi, f in enumerate(sorted(fs.glob(src.glob))):
        pf = pq.ParquetFile(fs.open(f))
        names = pf.schema_arrow.names
        cols = [c for c in names if c != "audio"]
        if src.dur is None:  # no duration column: measure it from the audio (only small dev sets need this)
            t = pf.read(columns=[*cols, "audio"]).to_pylist()
            durs = [sf.info(io.BytesIO(r["audio"]["bytes"])).duration for r in t]
        else:
            t = pf.read(columns=cols).to_pylist()
            durs = [src.dur(r) for r in t]
        for k, (r, d) in enumerate(zip(t, durs, strict=True)):
            rid = src.make_id(r, fi, k) if src.make_id else str(r[src.id_col])
            rows.append((f, k, rid, r[src.text_col] or "", float(d)))
    return rows


def fetch_audio(needed, tmp):
    """Write the audio bytes of `needed` {(file, row): uid} to tmp/uid, reading only the row groups that hold them."""
    by_file = {}
    for (f, k), uid in needed.items():
        by_file.setdefault(f, {})[k] = uid
    for f, rows in tqdm(by_file.items(), desc="fetch audio"):
        pf = pq.ParquetFile(fs.open(f))
        starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        groups = sorted({int(np.searchsorted(starts, k, side="right") - 1) for k in rows})
        for g in groups:
            col = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
            for k in range(starts[g], starts[g + 1]):
                if k in rows:
                    with open(f"{tmp}/{rows[k]}", "wb") as out:
                        out.write(col[k - starts[g]]["bytes"])


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def plan_chunks(ids, durs, chunk_key, rng):
    """Greedy-pack consecutive segments of each recording into chunks. Returns {key: [[row, ...], ...]}."""
    by_key = {}
    for row in sorted(range(len(ids)), key=lambda r: natural_key(ids[r])):
        by_key.setdefault(chunk_key(ids[row]), []).append(row)
    plans = {}
    for key, rows in by_key.items():
        chunks, cur, cur_dur, target = [], [], 0.0, rng.uniform(10, 28)
        for r in rows:
            add = durs[r] + (GAP if cur else 0.0)
            if cur and cur_dur + add > MAX_DUR:
                chunks.append((cur, cur_dur))
                cur, cur_dur, target, add = [], 0.0, rng.uniform(10, 28), durs[r]
            cur.append(r)
            cur_dur += add
            if cur_dur >= target:
                chunks.append((cur, cur_dur))
                cur, cur_dur, target = [], 0.0, rng.uniform(10, 28)
        if cur:
            chunks.append((cur, cur_dur))
        plans[key] = [c for c, d in chunks if d >= 10.0]  # a short leftover tail isn't a chunk
    return plans


def assign_splits(groups, rng):
    """Largest groups first, each into the split furthest below its clip target (ties broken randomly).

    Filling splits with groups in random order overshoots badly when a domain has only a few big groups
    (Earnings-22 has few calls, AMI few speakers); largest-first keeps every domain close to 70/15/15.
    """
    counts = Counter(groups)
    order = sorted(counts, key=lambda g: (-counts[g], rng.random()))
    split_of, filled, total = {}, Counter(), len(groups)
    for g in order:
        split_of[g] = max(FRACS, key=lambda s: FRACS[s] * total - filled[s])
        filled[split_of[g]] += counts[g]
    return [split_of[g] for g in groups]


def build_domain(name, dom, hours, per_key, rng, tmp):
    meta = [m for src in dom.sources for m in read_metadata(src)]
    seen, rows = set(), []
    for m in meta:  # drop unusable segments and duplicates across sources
        if 1.0 <= m[4] <= MAX_DUR and normalize(m[3]).strip() and m[2] not in seen:
            seen.add(m[2])
            rows.append(m)
    ids, durs = [m[2] for m in rows], np.array([m[4] for m in rows])
    plans = plan_chunks(ids, durs, dom.chunk_key, rng)

    # recordings in random order, a few chunks each (spread over the recording), until the hour budget
    chosen, total = [], 0.0
    for key in rng.permutation(sorted(plans)):
        chunks = plans[key]
        if len(chunks) > per_key:
            chunks = [chunks[i] for i in np.linspace(0, len(chunks) - 1, per_key).round().astype(int)]
        for c in chunks:
            chosen.append(c)
            total += durs[c].sum() + GAP * (len(c) - 1)
        if total >= hours * 3600:
            break
    groups = [dom.group_of(ids[c[0]]) for c in chosen]
    splits = assign_splits(groups, rng)

    os.makedirs(f"{tmp}/{name}", exist_ok=True)
    fetch_audio({(rows[r][0], rows[r][1]): f"{name}/{r}" for c in chosen for r in c}, tmp)
    silence = np.zeros(int(GAP * SR), dtype=np.float32)

    def gen():
        for c, group, split in zip(chosen, groups, splits, strict=True):
            arrays = [librosa.load(f"{tmp}/{name}/{r}", sr=SR)[0] for r in c]  # mono, resampled to 16 kHz
            wav = np.concatenate([x for a in arrays for x in (silence, a)][1:])
            yield {
                "audio": {"array": wav, "sampling_rate": SR},
                "text": " ".join(rows[r][3].strip() for r in c),
                "ID": f"{name}:{ids[c[0]]}+{len(c)}",
                "duration": len(wav) / SR,
                "source": name,
                "group": group,
                "split": split,
                "n_segments": len(c),
            }

    features = Features(
        {
            "audio": Audio(sampling_rate=SR),
            "text": Value("string"),
            "ID": Value("string"),
            "duration": Value("float64"),
            "source": Value("string"),
            "group": Value("string"),
            "split": Value("string"),
            "n_segments": Value("int64"),
        }
    )
    out = Dataset.from_generator(gen, features=features, cache_dir=f"{tmp}/hf_cache")
    out.save_to_disk(f"{OUT}/parts/{name}")
    del out
    shutil.rmtree(f"{tmp}/{name}")  # raw segment bytes and the generator cache are no longer needed
    shutil.rmtree(f"{tmp}/hf_cache", ignore_errors=True)
    c = Counter(splits)
    sg = Counter(g for g, s in zip(groups, splits, strict=True) if s == "train")
    print(
        f"{name:12s} {len(chosen):5d} chunks, {total / 3600:4.1f} h, {len(set(groups)):5d} groups "
        f"({len(sg)} train), {len(plans)} recordings available  "
        + "  ".join(f"{s} {c[s] / len(chosen):.0%}" for s in ("train", "val", "test")),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=20.0, help="audio budget per domain")
    ap.add_argument("--per-recording", type=int, default=8, help="max chunks from one recording")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--append", action="store_true", help="transcribe only --domains, add to existing splits")
    cli = ap.parse_args()
    OUT = cli.out
    tmp = f"{OUT}/tmp"

    for name in cli.domains:  # each domain is saved as it finishes, so a crash only loses the current one
        if not os.path.exists(f"{OUT}/parts/{name}"):
            build_domain(name, DOMAINS[name], cli.hours, cli.per_recording, np.random.default_rng(0), tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    ds = concatenate_datasets([load_from_disk(f"{OUT}/parts/{n}") for n in cli.domains])
    ds = score(ds, desc="speakers")
    for split in ("train", "val", "test"):
        part = ds.filter(lambda s, split=split: s == split, input_columns=["split"])
        if cli.append:  # existing split + the new domains; written aside, since the old one is memory-mapped
            old = load_from_disk(f"{OUT}/{split}_scored")
            part = concatenate_datasets([old, part.cast(old.features)])
            part.save_to_disk(f"{OUT}/{split}_scored.tmp")
            del old
            shutil.rmtree(f"{OUT}/{split}_scored")
            shutil.move(f"{OUT}/{split}_scored.tmp", f"{OUT}/{split}_scored")
            part = load_from_disk(f"{OUT}/{split}_scored")
        else:
            part.save_to_disk(f"{OUT}/{split}_scored")
        groups = len(set(part["group"]))
        print(split, len(part), "chunks,", groups, "groups", dict(sorted(Counter(part["source"]).items())))
