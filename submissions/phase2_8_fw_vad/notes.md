# phase2_8_fw_vad — faster-whisper VAD-on probe

**Decision:** failed; do not use VAD filtering for kirtan captions.

Overall: **25.4%** frame accuracy (`869/3425`), collar `1s`.

## Command

```bash
python3 scripts/run_path_a.py \
  --blend "token_sort_ratio:0.5,WRatio:0.5" \
  --threshold 0 \
  --stay-bias 6 \
  --blind \
  --blind-aggregate chunk_vote \
  --blind-lookback 30 \
  --live \
  --tentative-emit \
  --vad-filter \
  --out-dir submissions/phase2_8_fw_vad

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_8_fw_vad \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Interpretation

Enabling faster-whisper's Silero VAD drops most sung/instrumental kirtan regions. The run emits only `1-7` ASR chunks per source recording and many cold variants have zero post-lock segments.

This explains one part of the historical confusion: current faster-whisper default is `vad_filter=False`, and that is the correct setting for this domain. VAD-on is not a baseline recovery path.
