"""Compare the router vs. each single STT model on the held-out, speaker-disjoint test split."""

import argparse

import evaluate
import numpy as np
import pandas as pd
import torch

from common import DATA, error_matrix, load_router_splits, routed_errors, to_moonshine_features
from router_model import load_router
from transcribe import MODELS, normalize

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def route(model, ds, batch_size=32):
    """Index of the model the router picks for each clip -- computed from audio only."""
    picks = []
    for i in range(0, len(ds), batch_size):
        feats = {k: v.to(DEVICE) for k, v in to_moonshine_features(ds[i : i + batch_size]).items()}
        picks.append(model(**feats).logits.argmax(-1).cpu().numpy())
    return np.concatenate(picks)


def paired_bootstrap(err_a, err_b, n_words, iters=10_000, seed=0):
    """95% CI of corpus WER(a) - WER(b) under clip resampling, and P(a is not better than b)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n_words), size=(iters, len(n_words)))
    words = n_words[idx].sum(1)
    delta = err_a[idx].sum(1) / words - err_b[idx].sum(1) / words
    return np.percentile(delta, [2.5, 97.5]), (delta >= 0).mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/router")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--split", default="test", help="train shows whether the router can fit what it was shown")
    cli = ap.parse_args()
    model = load_router(cli.model).to(DEVICE).eval()
    labels = [model.config.id2label[k] for k in range(model.config.num_labels)]

    splits = load_router_splits(cli.data)
    test = splits[cli.split]
    choice = route(model, test)
    err = error_matrix(test, labels)
    df = test.remove_columns("audio").to_pandas()
    df["route"] = np.array(labels)[choice]
    df["err_router"] = routed_errors(choice, err)
    df["err_oracle"] = err.min(1)
    # per-source baseline: each source's best model on *train*, i.e. pure domain detection (needs source labels)
    train_err, train_src = error_matrix(splits["train"], labels), np.array(splits["train"]["source"])
    pick = {s: int(np.argmin(train_err[train_src == s].sum(0))) for s in np.unique(train_src)}
    df["err_per_source"] = routed_errors([pick[s] for s in df.source], err)

    # Headline corpus WER via 🤗 evaluate on the routed transcripts (cross-checks our per-clip counts).
    wer = evaluate.load("wer")
    refs = [normalize(t) for t in df.text]
    keep = [i for i, r in enumerate(refs) if r.strip()]
    hyps = np.stack([df[f"hyp_{m}"].to_numpy() for m in labels], 1)
    systems = {f"{m} ({MODELS[m][0].split('/')[-1]})": hyps[:, k] for k, m in enumerate(labels)}
    systems["router"] = hyps[np.arange(len(df)), choice]
    print(
        f"\n{cli.split.capitalize()} set: {len(df)} clips, {df.duration.sum() / 3600:.2f} h  |  routed to: "
        + ", ".join(f"{m} {(df.route == m).mean():.0%}" for m in labels)
        + "\n"
    )
    for name, h in systems.items():
        score = wer.compute(references=[refs[i] for i in keep], predictions=[normalize(h[i]) for i in keep])
        print(f"  {name:42s} WER {score:.2%}")
    print(f"  {'per-source (domain detection, needs labels)':42s} WER {df.err_per_source.sum() / df.n_words.sum():.2%}")
    print(f"  {'oracle (per-clip ceiling)':42s} WER {df.err_oracle.sum() / df.n_words.sum():.2%}")

    # the goal: how much of the best-single -> oracle gap is closed
    single, oracle = err.sum(0).min(), df.err_oracle.sum()
    print(
        f"\nOracle gap closed: router {(single - df.err_router.sum()) / (single - oracle):.0%}, "
        f"per-source {(single - df.err_per_source.sum()) / (single - oracle):.0%}"
    )

    n = df.n_words.to_numpy()
    for k, m in enumerate(labels):
        ci, p_val = paired_bootstrap(df.err_router.to_numpy(), err[:, k], n)
        print(f"router - {m:10s}: 95% CI [{ci[0]:+.2%}, {ci[1]:+.2%}]  bootstrap p={p_val:.4f}")

    by_src = df.groupby("source").apply(
        lambda g: pd.Series(
            {"clips": len(g)}
            | {m: g[f"err_{m}"].sum() / g.n_words.sum() for m in labels}
            | {"per_src": g.err_per_source.sum() / g.n_words.sum()}
            | {"router": g.err_router.sum() / g.n_words.sum(), "oracle": g.err_oracle.sum() / g.n_words.sum()}
            # router minus per-source pick: only per-clip routing can make this negative
            | {"within_gain": (g.err_router.sum() - g.err_per_source.sum()) / g.n_words.sum()}
            | {f"%->{m}": (g.route == m).mean() for m in labels}
        ),
        include_groups=False,
    )
    print("\nPer-source WER (source is never shown to the router):")
    print(by_src.to_string(float_format=lambda x: f"{x:.3f}"))
