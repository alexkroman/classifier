# stt-router

A tiny audio classifier that listens to a clip (audio only, no transcript) and routes it to whichever
of three small English STT models will make the fewest word errors:

| model | params | type |
|---|---|---|
| [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) | 27M | encoder-decoder |
| [`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en) | 38M | encoder-decoder |
| [`nvidia/stt_en_conformer_ctc_small`](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) | 13M | NeMo CTC |

**Goal metric:** the share of the gap between the best single model and the per-clip oracle (always
picking the best model for each clip) that the router closes. **Secondary reference:** a per-source
baseline that picks each dataset's best model. It needs dataset labels, so it amounts to pure domain
detection.

## Pipeline

```bash
uv sync
uv run python prepare_speakers.py                            # build data/speakers: download, chunk, transcribe with all 3 models
uv run python prepare_speakers.py --domains edacc --append   # how EdAcc was added (optional)

# LP-FT, stage 1: frozen Moonshine encoder, train the head only
uv run python train_router.py --epochs 8 --out models/router_lp
# stage 2: unfreeze the encoder, starting from the stage-1 head, with SpecAugment
uv run python train_router.py --init-from models/router_lp --finetune-encoder --encoder-lr 1e-5 --lr 1e-4 \
    --epochs 5 --specaug-time 0.1 --specaug-feat 0.05 --out models/router

uv run python evaluate_router.py --model models/router                 # test split (default)
uv run python evaluate_router.py --model models/router --split train   # fit check
```

| file | role |
|---|---|
| `prepare_speakers.py` | builds the dataset (download, chunking, splits, transcription) |
| `transcribe.py` | the 3 STT models; Whisper normalizer + jiwer word errors |
| `common.py` | splits, targets, Moonshine feature extraction |
| `router_model.py` | `MoonshineRouter` |
| `train_router.py` | training |
| `evaluate_router.py` | evaluation |

`train_router.py` flags: `--data` (default `data/speakers`), `--init-from`, `--finetune-encoder`,
`--lr` (head, default 1e-3), `--encoder-lr` (1e-5), `--epochs` (8), `--head-hidden` (256),
`--huber-delta` (5.0), `--specaug-time`, `--specaug-feat`, `--out`.
`evaluate_router.py` flags: `--model`, `--data`, `--split`.

## How it works

**Data (`data/speakers`).** Only held-out splits are used, so none of the candidate models trained on
this audio: the ESB test sets (via `hf-audio/open-asr-leaderboard`), plus AMI IHM dev, LibriSpeech dev,
VoxPopuli dev, all 125 Earnings-22 calls (via `distil-whisper/earnings22`), SPGISpeech test, GigaSpeech
test, and EdAcc dev+test. Consecutive segments from one recording and speaker are packed into 10-29.5 s
chunks, which fit Whisper's 30 s window. Each recording contributes at most 8 chunks, spread across it,
to get more distinct voices per hour. Every chunk is re-transcribed by all 3 models. Splits are 70/15/15
per domain by speaker/recording (EdAcc by conversation), so no voice appears in two splits. Parquet
files are read remotely with column projection, which avoids filling the HF cache.

| domains | hours | train | val | test | training speaker groups |
|---|---|---|---|---|---|
| 7 | ~67 | 8,069 | 1,732 | 1,736 | 793 |

*Why chunks:* on short clips (the earlier `data/mix`, median ~7 s), 23% of clips were 3-way ties and
26% differed by a single word, so the labels were mostly noise. At 10-30 s, ties drop to 7%.

**Model (`MoonshineRouter`).** The moonshine-tiny encoder (7.7M params) takes the raw 16 kHz waveform
at its true length, with no padding to 30 s. Batches are padded to multiples of 2 s because MPS leaks a
compiled graph for every distinct input shape. On top of the encoder: learned softmax weights over all
encoder layers, a linear projector, masked mean pooling over real frames, a GELU hidden layer (256),
and 3 outputs. The outputs are the *negated predicted cost* of each STT model, and the route is the
argmax.

**Loss.** Huber regression on each model's word errors minus the clip's mean over the 3 models
(difference regression). Regression recovers conditional means, so the argmax is the Bayes rule
argmin_k E[errors_k | audio]. Centering cancels the difficulty shared by all models on a clip, and no
cap is needed. Tried and rejected:

- expected-regret loss: consistent, but its gradient vanishes; it collapsed to always choosing conformer.
- soft cross-entropy against softmax(-regret/τ): consistent only as τ→∞.
- regressing log1p(errors): biased by Jensen's inequality.

**Training (LP-FT).** Stage 1 trains the head on the frozen encoder. Stage 2 unfreezes the encoder
(LR 1e-5, head 1e-4) and starts from the stage-1 head. Plain fine-tuning with a randomly initialized
head distorted the pretrained features and scored below the frozen baseline. In stage 2, SpecAugment-
style masking on the conv front end's output (time spans and channel bands, training only) acts as a
regularizer. The saved model is the last epoch of each stage; validation routed WER is logged every
epoch for monitoring but not used to pick a checkpoint, so val noise doesn't leak into the model choice.

**Evaluation.** The test split contains only unseen speakers. The script reports corpus WER
(cross-checked with Hugging Face `evaluate`), paired-bootstrap CIs against each single model, the share of the
oracle gap closed, and a per-source table with `within_gain`: router WER minus the per-source pick's WER
for each dataset. Only per-clip routing can make it negative.

## Results

**Previous LP-FT run.** Same recipe *without* SpecAugment and with a 4-epoch head, run *before* EdAcc
was added. Test: 1,594 chunks, 9.3 h. WER (%):

| dataset | moonshine | whisper | conformer | best per dataset (picked on train) | router | oracle |
|---|---|---|---|---|---|---|
| ami | 21.10 | 21.79 | 17.27 | 17.27 | 17.34 | 15.16 |
| earnings22 | 24.64 | 25.84 | 26.40 | 25.84 | 25.31 | 19.21 |
| gigaspeech | 11.66 | 11.82 | 13.73 | 11.82 | 11.42 | 8.88 |
| librispeech | 7.70 | 8.62 | 5.73 | 5.73 | 5.71 | 4.68 |
| spgispeech | 7.92 | 7.10 | 9.21 | 7.10 | 7.25 | 4.98 |
| voxpopuli | 12.32 | 11.49 | 9.28 | 9.28 | 9.45 | 8.20 |
| **ALL** | 11.50 | 11.55 | 11.06 | 10.11 | **10.08** | 7.92 |

Router vs. the best single model (conformer): -0.98 WER points, 95% CI [-1.38, -0.49], p=0.0001.
The router closes 31% of the oracle gap; the per-source baseline closes 30%.

**Final run (EdAcc + SpecAugment).** The recipe in [Pipeline](#pipeline): 8-epoch head, then 5 epochs of
fine-tuning with SpecAugment (time 0.1, channels 0.05), last epoch of each stage
(`models/router_speakers_moon_lpft_sa`). Test: 1,736 chunks, 10.1 h, all speakers unseen. WER (%):

| dataset | moonshine | whisper | conformer | best per dataset (picked on train) | router | oracle |
|---|---|---|---|---|---|---|
| ami | 21.1 | 21.8 | 17.3 | 17.3 | 17.3 | 15.2 |
| earnings22 | 24.6 | 25.8 | 26.4 | 25.8 | 25.4 | 19.2 |
| edacc | 36.7 | 36.9 | 34.5 | 34.5 | 34.2 | 30.7 |
| gigaspeech | 11.7 | 11.8 | 13.7 | 11.8 | 12.1 | 8.9 |
| librispeech | 7.7 | 8.6 | 5.7 | 5.7 | 5.7 | 4.7 |
| spgispeech | 7.9 | 7.1 | 9.2 | 7.1 | 7.1 | 5.0 |
| voxpopuli | 12.3 | 11.5 | 9.3 | 9.3 | 9.5 | 8.2 |
| **ALL** | 13.36 | 13.42 | 12.78 | 11.91 | **11.91** | 9.60 |

Router vs. the best single model (conformer): -0.87 WER points, 95% CI [-1.26, -0.42], p=0.0006; it is
also significantly better than moonshine and whisper. It routes 51% to conformer, 44% to whisper and 5%
to moonshine. **On test it ties the per-source baseline exactly** (both close 28% of the oracle gap):
small within-source gains on earnings22 and EdAcc (-0.4 points each) are cancelled by small losses on
gigaspeech and voxpopuli (+0.2 each).

Fit check, % of the oracle gap closed (stage 1 = the frozen-encoder head the fine-tune starts from):

| | train | test | test WER |
|---|---|---|---|
| per-source baseline | 36% | 28% | 11.91 |
| stage 1 (frozen encoder, 8-epoch head) | 37% | 28% | 11.90 |
| stage 2 (fine-tuned encoder + SpecAugment) | 39% | 28% | 11.91 |

Fine-tuning raises the training fit by 2 points over stage 1 but transfers nothing to unseen speakers,
even with SpecAugment. On this data the router is a domain detector: it beats every single model, and
matches but does not beat routing by dataset label.

## Findings

- **The router mostly learns domain detection.** A linear probe recovers the source from frozen
  Moonshine features with 96.5% accuracy, while source explains only 4.2% of per-chunk cost variance.
  An MSE-style regressor therefore ends up predicting each domain's mean cost vector.
- **The remaining headroom is speaker-level and chunk-level noise.** A per-speaker hindsight oracle
  closes ~57% of the gap. Frozen ASR features can't express speaker identity, because ASR encoders are
  trained to ignore speaker and channel. Fine-tuning fits the training speakers (+3 points over
  per-source on train in the final run) but transfers nothing to test (0 points), with or without
  SpecAugment. More distinct speakers is the lever: there are only 793 training voices.
- **Moonshine is rarely chosen.** It is never the best on average in any training domain. It is
  strictly best on only 16% of chunks, by a median of 1 word, and loses on 60%, by a median of 2.
- **Headroom per domain** is largest where error rates are high, the models are close on average, and
  their errors are uncorrelated (earnings22: 5.2 points of room, error correlation 0.52; gigaspeech:
  models tied, 3.0 points). It is smallest where one model dominates or all of them fail together
  (EdAcc: error correlation 0.98; conformer is best for 15 of 17 accents).
- **Next levers:**
  - more speakers: pseudo-label untranscribed audio with a much stronger model (e.g. Universal-3.5
    Pro), or use VoxCeleb.
  - denser, time-aligned error targets.
  - augmentation with re-transcription: speed, noise, and reverb change the label, so augmented
    audio has to be transcribed again.
