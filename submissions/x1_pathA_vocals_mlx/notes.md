# x1_pathA_vocals_mlx — Demucs vocal separation + mlx-large-v3

Phase X1 probe: pre-process audio with Demucs (`htdemucs`) to isolate vocals
before transcription. Compares to the raw-audio mlx baseline (`v4_mlx_large_v3`).

## Score: 82.6% (+10.5 over raw mlx, still -3.9 below Path A v3.2 fw canonical)

## Per-case: raw mlx → vocals mlx

| Case | raw | vocals | Δ |
|---|---|---|---|
| IZOsmkdmmcg | 80.0 | 87.7 | +7.7 |
| IZOsmkdmmcg_cold33 | 6.2 | **82.8** | **+76.6** |
| IZOsmkdmmcg_cold66 | 83.3 | 80.8 | -2.5 |
| kZhIA8P6xWI | 96.7 | 84.5 | -12.2 |
| kZhIA8P6xWI_cold33 | 92.8 | 83.6 | -9.2 |
| kZhIA8P6xWI_cold66 | 88.6 | 73.3 | -15.3 |
| kchMJPK9Axs | 82.9 | 90.3 | +7.4 |
| kchMJPK9Axs_cold33 | 78.3 | 81.3 | +3.0 |
| kchMJPK9Axs_cold66 | 82.0 | 92.3 | +10.3 |
| zOtIpxMT9hU | 75.3 | 78.4 | +3.1 |
| zOtIpxMT9hU_cold33 | 8.2 | **69.9** | **+61.7** |
| zOtIpxMT9hU_cold66 | 83.8 | 35.4 | -48.4 |

The +76.6 and +61.7 wins are blind-ID rescues — vocal separation cleared up
shabad-ID confusion for two cases that were failing entirely on raw audio.

## Run

```bash
python scripts/separate_vocals.py
python scripts/run_path_a.py --backend mlx_whisper --model large-v3 \
  --audio-dir audio_vocals --asr-cache-dir asr_cache_vocals \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 45 \
  --live --tentative-emit \
  --out-dir submissions/x1_pathA_vocals_mlx
```
