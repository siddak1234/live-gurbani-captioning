# live-gurbani-captioning

Engine for the [Live Captioning for Gurbani Kirtan benchmark](../live-gurbani-captioning-benchmark-v1/). Stage 0 is a measurement loop that emits empty submissions (no segments) and scores the ~26% floor; subsequent stages add audio download + BaniDB corpus caching, then a real Whisper + fuzzy-match engine with oracle and offline modes.

## Usage

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py                      # writes submissions/v0_empty/
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v0_empty/ \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/
```
