# phase2_9_score_lattice — post-lock evidence inspection

**Purpose:** first Phase 2.9.A diagnostic. This is not a submission and does
not produce a benchmark score. It inspects the post-lock score lattice that a
future loop-aware aligner will consume.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/dump_score_lattice.py \
  --out-dir diagnostics/phase2_9_score_lattice \
  --adapter-dir lora_adapters/v5b_mac_diverse
```

## Aggregate result

| Metric | Count |
|---|---:|
| Cases | 12 |
| Post-lock ASR chunks | 314 |
| Chunks whose midpoint overlaps GT lyric | 268 |
| Local best line equals GT | 235 / 268 |
| Stay-bias chosen line equals GT | 244 / 268 |
| Chunks whose midpoint has no GT lyric | 46 |

The local matcher evidence is much stronger than the final submission score
suggests: stay-bias agrees with GT on **91.0%** of chunks that overlap a GT
lyric. The remaining error is mostly not "the ASR cannot identify the line."
It is alignment/timing/null handling:

- chunks with no GT lyric still get forced into some canonical line;
- short cold-start cases have very few post-lock chunks, so a one-chunk timing
  error has a large frame-level penalty;
- legal refrain/rahao loops need explicit structure, not a generic
  line-distance penalty.

## Case highlights

| Case | Chunks with GT | Stay-bias matches GT | Read |
|---|---:|---:|---|
| `zOtIpxMT9hU_cold66` | 3 | 2 | Worst score case, but local best is 3/3. Failure is boundary/null timing, not recognition. |
| `kchMJPK9Axs_cold66` | 16 | 16 | Local evidence is perfect, yet Viterbi smoothing regressed it; do not over-regularize loops. |
| `kZhIA8P6xWI_cold33` | 17 | 14 | Some local ambiguity remains, but not enough to explain the whole 86.6% ceiling. |

## Decision

Proceed with a loop-aware aligner over this score lattice. Its first job is not
better per-chunk classification; it is turning already-good chunk evidence into
better segment boundaries while allowing null/background spans and legal line
returns.
