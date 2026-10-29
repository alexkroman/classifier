# stt-router

A tiny audio classifier that looks at a clip and decides which of two small English STT models
will transcribe it with fewer word errors:

- [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny)
- [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en)

The router is the `openai/whisper-tiny` encoder (frozen) with a learned weighted sum over its layers,
masked mean-pooling over the real (non-padding) frames, and a linear head. It only ever sees audio.

## Pipeline

```bash
uv sync
uv run python prepare_data.py        # 10k train clips (10 Loquacious shards) + full Loquacious test split
uv run python transcribe.py          # transcribe everything with both models, store per-clip word errors
uv run python train_router.py --out models/router_frozen            # frozen encoder (default)
uv run python evaluate_router.py --model models/router_frozen       # test-set comparison
```

Optional: `train_router.py --finetune-encoder --lr 3e-5` fine-tunes the encoder too, and
`evaluate_router.py --tune-threshold` picks the P(whisper) cutoff on validation instead of using 0.5.

## How it works

- **Data**: [Loquacious Set](https://huggingface.co/datasets/speechbrain/LoquaciousSet) mixes read
  speech (LibriLight), crowd-sourced (Common Voice), parliament (VoxPopuli), and noisy web audio
  (People's Speech, YODAS), so each model wins on different clips. Clips are limited to 1–30 s.
- **Targets**: `label` = whichever model made fewer word errors (Whisper normalizer + jiwer);
  `weight` = how many errors that choice saves, capped at 10. Ties (~42% of clips) get weight 0.
  The loss is weighted cross-entropy, so the router optimizes errors saved, not accuracy.
- **Validation** is speaker-/recording-disjoint (`common.load_router_splits`). With a random split,
  voices leak across the boundary and validation scores come out too optimistic.
- **Model selection** uses routed corpus WER on validation, not loss.
- **Evaluation** runs on the official Loquacious test split (speakers never seen in training), with
  WER from 🤗 `evaluate`, paired-bootstrap CIs against each single model, and a per-source breakdown.

## Results

Frozen-encoder router (`models/router_frozen`, 8 epochs), Loquacious test split: 8,069 clips, 16.4 h.

| system | test WER |
|---|---|
| moonshine-tiny | 15.80% |
| whisper-tiny.en | 16.17% |
| **router** (P(whisper) ≥ 0.5, sends 61% to Whisper) | **15.80%** |
| best model per source (needs source labels) | 14.85% |
| per-clip oracle (ceiling) | 11.82% |

Router vs. Moonshine: 95% CI [−0.36%, +0.31%], so **no improvement**. Router vs. Whisper: −0.37%, p=0.004.
On validation the router scored 10.20% against Moonshine's 11.18%, but that gain did not carry over to test.
Tuning the threshold on validation (0.48) made test slightly worse (15.90%).

**Diagnosis.** The router only acts as a source/domain detector. On contested clips (no tie), its AUC is
0.64 on validation but just 0.535 on test. *Within* any single source the AUC is about 0.5 on both, so it
cannot tell which model wins clip-to-clip. Even the domain detection fails to transfer to test: it sends
only 15% of YODAS to Whisper, though Whisper is better there. Training uses just 2 shards per source,
which is likely too few recordings for the domain cues to generalize.
