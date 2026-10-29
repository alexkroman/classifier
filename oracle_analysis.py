"""How much could routing between STT models gain, and how much of that is learnable?

For every subset of >= 2 scored models, report on a scored split:
  best single   one model for everything
  per-source    best model chosen per source corpus -- the part a router can get just by recognizing conditions
  per-clip      best model chosen per clip -- the (optimistic) ceiling
"""

import argparse
from itertools import combinations

import numpy as np
from datasets import load_from_disk


def wer(err, n):
    return err.sum() / n.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    cli = ap.parse_args()

    ds = load_from_disk(f"data/{cli.split}_scored").remove_columns("audio")
    models = [c[4:] for c in ds.column_names if c.startswith("err_")]
    err = {m: np.array(ds[f"err_{m}"]) for m in models}
    n, src = np.array(ds["n_words"]), np.array(ds["source"])
    sources = np.unique(src)

    print(f"{cli.split}: {len(n)} clips\n\nPer-source WER:")
    print(f"{'source':16s}{'clips':>7s}" + "".join(f"{m:>11s}" for m in models) + f"{'oracle':>11s}")
    for s in sources:
        k = src == s
        oracle = np.min([err[m][k] for m in models], 0).sum() / n[k].sum()
        print(f"{s:16s}{k.sum():7d}" + "".join(f"{wer(err[m][k], n[k]):11.2%}" for m in models) + f"{oracle:11.2%}")
    print(
        f"{'ALL':16s}{len(n):7d}"
        + "".join(f"{wer(err[m], n):11.2%}" for m in models)
        + f"{np.min([err[m] for m in models], 0).sum() / n.sum():11.2%}"
    )

    print(f"\n{'models':32s}{'best single':>13s}{'per-source':>12s}{'per-clip':>10s}{'clip gain':>11s}")
    for r in range(2, len(models) + 1):
        for subset in combinations(models, r):
            single = min(wer(err[m], n) for m in subset)
            per_src = sum(min(err[m][src == s].sum() for m in subset) for s in sources) / n.sum()
            per_clip = wer(np.min([err[m] for m in subset], 0), n)
            gain = (single - per_clip) / single
            print(f"{' + '.join(subset):32s}{single:13.2%}{per_src:12.2%}{per_clip:10.2%}{gain:11.0%}")
