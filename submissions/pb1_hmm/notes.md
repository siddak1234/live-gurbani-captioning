# pb1_hmm — Path B Phase B3 first end-to-end run

**Result: 61.0% overall. Significantly below Path A v3.2's 86.5%.**

## Config

- ASR: MMS-1B with Punjabi (`pan`) language adapter, ~50Hz frame rate, full
  audio encoded in 60s chunks on Apple Silicon GPU (MPS).
- Decoder: `ShabadHmm` forward algorithm. States = one macro state per
  shabad line, each containing a standard CTC alignment lattice for its
  Gurmukhi character sequence. Cross-line transitions: any line end → any
  line start with `switch_log_prob = log(1e-4) = -9.2`.
- Per-frame prediction: `argmax` over lines of the marginal-summed alpha.
- Skip line 0 (shabad title) at HMM build time (matches Path A convention).
- Oracle (GT shabad_id given), offline.

## Run

```bash
python scripts/run_path_b_hmm.py --out-dir submissions/pb1_hmm
```

## Per-shabad vs Path A v3.2

| Shabad | Path A v3.2 | Path B HMM | Δ |
|---|---|---|---|
| IZOsmkdmmcg | 98 / 96 / 94 | 72 / 72 / 62 | -22 to -33 |
| kZhIA8P6xWI | 87 / 81 / 88 | 70 / 79 / 90 | -2 to -17 |
| kchMJPK9Axs | 83 / 77 / 76 | 54 / 53 / 83 | -29 to +7 |
| zOtIpxMT9hU | 93 / 79 / 83 | 43 / 32 / 19 | **-50 to -64** ✗ |

The zOtIpxMT9hU collapse is the worst symptom: prediction stuck on long monolithic stretches of wrong lines, with later parts of the audio scoring worse (-50 → -64 across cold variants).

## What I tried

**Sweep on `switch_log_prob`** — values -3, -4, -5, -6, -8 all scored 59-62%. Not the lever.

**Viterbi (max-paths) instead of forward (sum-paths)** — same range, 57-60%. Not the issue.

## Diagnosis

The HMM math is correct, but the underlying signal is too weak for frame-level line discrimination on slow kirtan:

1. **CTC blank-dominance**: MMS emits the blank token at ~99% of frames during sustained singing. Most frames carry near-zero discriminative information about which line is being sung — the blank fits everywhere in every line's lattice. The actual character events that distinguish lines are sparse (maybe 1 per 0.5-1s of singing for slow kirtan tempo).

2. **Within-line alignment freedom**: each line's CTC lattice can place its sparse character emissions wherever in the frame range they best fit. With long, sustained sounds, multiple lines look plausible — the lattice is flexible by design.

3. **Forward marginalizes across positions**: longer lines have more states, more paths through the lattice, and more accumulated alpha mass — even before character events. Viterbi (best-path) doesn't help because it picks the longest plausible path, also length-biased.

Path A's coarse text-level matching may be doing something better than fine-grained frame-level alignment for this task: Whisper's chunk-level transcripts implicitly capture the *sequence* of characters with relative timing, and rapidfuzz matching on those sequences is more discriminative than CTC's allow-anywhere alignment.

## Where this leaves Path B

The B0-B3 build (encoder, tokenizer, ctc_scorer, hmm) is real reusable scaffolding, but the immediate frame-level decoding strategy doesn't outperform Path A. Possible escape hatches if we want to keep pursuing this:

1. **Forced alignment over the full shabad** — instead of "HMM over lines as macro states", treat the full shabad's concatenated text as one long target with allowed loops/repetition. Aligns once over the entire audio. More principled but requires loop-aware lattice construction.
2. **Fine-tune MMS on Gurbani** — collect labeled kirtan audio, fine-tune. Real ML work, requires GPU time and labeled data.
3. **Different acoustic model** — maybe a phoneme-level CTC tuned for sung material exists. Worth searching for.
4. **Multi-resolution scoring** — combine frame-level CTC scores with chunk-level rapidfuzz scores. Hybrid Path A + B.

None of these is a quick win.

## Honest call

Path B at this point is a real engineering exercise with a working implementation that *underperforms the simpler Path A*. The "92-95% ceiling" I projected earlier was based on the *general* CTC+HMM literature, not on the specific behavior of MMS on sung Punjabi audio. The 60% result suggests the foundation model needs to be much better adapted to kirtan before the principled architecture pays off.

## Artifacts

- 12 submission JSONs in this directory
- The B0-B3 scaffold lives in `src/path_b/` — encoder, tokenizer, ctc_scorer, hmm
- Reproducer: `python scripts/run_path_b_hmm.py --out-dir submissions/pb1_hmm`
