# x6_ensemble — 3-way per-shabad ensemble (92.8%)

**Score: 92.8%** — up from X5's 91.2% (2-way) and v3.2's 86.5%.

## Route table

v3.2's blind shabad ID (12/12 correct) is the dispatcher.

| Predicted shabad | Engine used | Why |
|---|---|---|
| 4377 (IZOsmkdmmcg) | v3.2 (fw-medium) | v3.2 wins all 3 variants |
| **1821 (kZhIA8P6xWI)** | **v4 mlx-large-v3** | **mlx wins by +1-12 pts on all 3 variants** |
| 1341 (kchMJPK9Axs) | x4 surt-small-v3 | surt wins by +10-17 pts on all 3 |
| 3712 (zOtIpxMT9hU) | v3.2 | v3.2 wins 2 of 3 (cold33 only) |

## Per-shabad results

| Shabad | Cold0 | Cold33 | Cold66 | Source |
|---|---|---|---|---|
| IZOsmkdmmcg | 98.2 | 96.1 | 94.2 | v3.2 |
| **kZhIA8P6xWI** | **96.7** | **92.8** | **88.6** | **mlx-large-v3** |
| kchMJPK9Axs | 92.0 | 92.5 | 92.8 | surt |
| zOtIpxMT9hU | 93.0 | 79.1 | 82.8 | v3.2 |

## Run

```bash
python scripts/ensemble_submissions.py
```

(Needs `v3_2_pathA_no_title`, `x4_pathA_surt`, `v4_mlx_large_v3` to exist.)

## Where the remaining 2.2 points to 95% live

Mostly **zOtIpxMT9hU cold33 (79.1%)** and **cold66 (82.8%)** — the audio has rapid line transitions (L4-L5-L6-L5-L6 over short windows) that every engine struggles with. mlx wins cold66 here marginally (83.8%) but loses cold33 catastrophically (8.2% blind ID failure), so case-level routing without GT-aware metadata isn't an obvious win.

Real paths to 95%:
1. **Custom training on rapid-transition kirtan** (multi-week, real ML)
2. **Forced alignment over the full shabad** — Path B done right (architectural rebuild)
3. **Per-frame intra-case ensembling** — blend engines' frame-level predictions probabilistically (real engineering build)

None of these are quick wins; X6 may be the stopping point for "fast ensemble of pre-trained components."
