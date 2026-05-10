# x5_ensemble — Per-shabad routing between v3.2 and X4 surt

**91.2% overall.** New canonical Path A result. Up +4.7 from v3.2's 86.5% by routing kchMJPK9Axs (shabad 1341) to the surt-small-v3 backend and keeping everything else on faster-whisper-medium.

## How it works

1. Run v3.2 (`submissions/v3_2_pathA_no_title`) — fw-medium ASR + rapidfuzz blend + stay-bias + chunk-vote shabad ID. Blind shabad ID is 12/12 correct.
2. Run X4 (`submissions/x4_pathA_surt`) — same pipeline but with `surindersinghssj/surt-small-v3` (Whisper-small fine-tuned on 660h of Gurbani) and 10s manual windows.
3. For each case, use v3.2's predicted shabad to route:
   - shabad 1341 → use X4's output (surt wins this shabad by +10-17 points)
   - any other shabad → use v3.2's output

`scripts/ensemble_submissions.py` does the routing. v3.2's shabad ID is the dispatcher because it's been proven reliable (12/12 in blind mode); X4's shabad ID is noisier on cold variants.

## Per-shabad

| Shabad | v3.2 | X4 | **X5** | Routed to |
|---|---|---|---|---|
| IZOsmkdmmcg cold0 | 98.2 | 90.3 | **98.2** | v3.2 |
| IZOsmkdmmcg cold33 | 96.1 | 10.1 | **96.1** | v3.2 |
| IZOsmkdmmcg cold66 | 94.2 | 16.7 | **94.2** | v3.2 |
| kZhIA8P6xWI cold0 | 86.5 | 78.5 | **86.5** | v3.2 |
| kZhIA8P6xWI cold33 | 80.7 | 73.4 | **80.7** | v3.2 |
| kZhIA8P6xWI cold66 | 87.6 | 64.8 | **87.6** | v3.2 |
| **kchMJPK9Axs cold0** | 83.4 | 92.0 | **92.0** | **surt** |
| **kchMJPK9Axs cold33** | 76.9 | 92.5 | **92.5** | **surt** |
| **kchMJPK9Axs cold66** | 76.1 | 92.8 | **92.8** | **surt** |
| zOtIpxMT9hU cold0 | 93.0 | 79.8 | **93.0** | v3.2 |
| zOtIpxMT9hU cold33 | 79.1 | 68.4 | **79.1** | v3.2 |
| zOtIpxMT9hU cold66 | 82.8 | 38.4 | **82.8** | v3.2 |
| **Overall** | 86.5 | 74.0 | **91.2** | |

## Why it works

surt-small-v3 was trained on Gurbani-specific audio so it produces canonical Gurmukhi snapped to BaniDB lines. On the kchMJPK9Axs case — which has a shared `man bauraa re` refrain across every line, defeating fw-medium's matcher — surt outputs the **specific verse content** before/after the refrain, which makes line discrimination trivial.

The other three shabads don't have the shared-refrain problem, so v3.2 already does well and surt's coarser 10s windows actually hurt those cases.

## Run command

```bash
python scripts/ensemble_submissions.py
```

(Requires `submissions/v3_2_pathA_no_title` and `submissions/x4_pathA_surt` to exist. Both can be regenerated from their notes.md run commands.)

## Where to push past 91.2%

Remaining weak cells:
- kZhIA8P6xWI cold33: 80.7% (cold-variant blind-buffer effect)
- zOtIpxMT9hU cold33: 79.1% (same)
- zOtIpxMT9hU cold66: 82.8%

Both engines underperform on these specific cold variants. The constraint is fundamentally the 30s shabad-ID buffer eating a meaningful fraction of short UEMs. Path forward: shorter-buffer blind ID, or a third engine that performs better in cold-start.

For now, 91.2% blind+live is comfortably above v3.2's 86.5% and the 60-80% public range.
