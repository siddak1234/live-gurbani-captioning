# x1_pathA_vocals_fw — Demucs vocal separation + faster-whisper-medium

Canonical Path A v3.2 pipeline, transcribing Demucs-isolated vocals
instead of raw audio.

## Score: 66.0% (-20.5 from v3.2's 86.5%)

## What failed

Three of `kZhIA8P6xWI`'s blind IDs collapsed:

```
  kZhIA8P6xWI:        predicts 1341 (GT 1821) ✗
  kZhIA8P6xWI_cold33: predicts 3712 (GT 1821) ✗
  kZhIA8P6xWI_cold66: predicts 4377 (GT 1821) ✗
```

Compare to v3.2 on raw audio: all three correctly IDed shabad 1821.

## Asymmetric outcome

Same Demucs-isolated vocals, two different ASR backends:

| Backend | Raw audio | Isolated vocals |
|---|---|---|
| faster-whisper-medium | 86.5% (v3.2) | **66.0%** (-20.5) |
| mlx-whisper large-v3 | 72.1% | 82.6% (+10.5) |

Vocal separation **hurts fw, helps mlx**. Probably because Demucs introduces
artifacts (clipping, phase smearing) that interact badly with fw's Silero VAD
pre-filter or its decoder defaults. mlx's lighter ingestion handles the
artifacts better.

## Implication

Vocal source separation isn't a free lift. The lyrics-alignment literature
reports gains in clean studio recordings; kirtan's vocal-instrumental
balance (sangat singing along with the ragis, harmonium overlapping the
human voice register) is messier and the separator's errors are worse.

For Path A canonical, **stick with raw audio** until we have either a
fine-tuned ASR (Phase X3) or a better separator that doesn't degrade fw input.

## Run

```bash
python scripts/separate_vocals.py
python scripts/run_path_a.py \
  --audio-dir audio_vocals --asr-cache-dir asr_cache_vocals \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 30 \
  --live --tentative-emit \
  --out-dir submissions/x1_pathA_vocals_fw
```
