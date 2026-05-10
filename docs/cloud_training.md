# Cloud training guide — Phase B fine-tune

Local Mac training works for smoke tests but isn't practical for 20+ hours of audio. Use cloud GPU instead. Two recommended options:

## Option 1: Google Colab (easiest, free tier works for small data)

Free Colab gives you a T4 GPU (~16 GB VRAM). Pro/Pro+ gives A100s. For LoRA fine-tuning on w2v-bert (600M base, 3.3M trainable), free T4 is enough for ~10 hours of audio per session; A100 will handle 50+ hours.

### Cell-by-cell setup

Open a new Colab notebook, set runtime to **GPU**, then paste these cells in order.

**Cell 1 — Install dependencies**
```python
!pip install -q faster-whisper rapidfuzz unidecode yt-dlp transformers \
    peft accelerate datasets soundfile mlx-whisper torchcodec
```

**Cell 2 — Mount Google Drive (for persistent storage)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3 — Clone your repo**
```python
%cd /content
!git clone https://github.com/<your-username>/live-gurbani-captioning.git
%cd live-gurbani-captioning
```

**Cell 4 — Pull your training manifest from Drive**
Place your manifest + clips directory under `/content/drive/MyDrive/kirtan_training/v1/`.
```python
import shutil, pathlib
src = pathlib.Path("/content/drive/MyDrive/kirtan_training/v1")
dst = pathlib.Path("/content/live-gurbani-captioning/training_data/kirtan_v1")
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists():
    shutil.copytree(src, dst)
print(f"manifest: {(dst / 'manifest.json').exists()}")
print(f"clips: {len(list((dst / 'clips').glob('*.wav')))} files")
```

**Cell 5 — Run fine-tune (CUDA, no MPS fallback needed)**
```python
!python scripts/finetune_path_b.py \
    --manifest training_data/kirtan_v1/manifest.json \
    --output-dir /content/drive/MyDrive/kirtan_training/lora_v1 \
    --model-id kdcyberdude/w2v-bert-punjabi \
    --epochs 3 --batch-size 4 --grad-accum 4 \
    --lr 5e-5 --warmup-steps 200 --save-steps 500
```

The adapter is saved directly to Drive so the Colab session timeout doesn't lose it.

**Cell 6 — Evaluate against benchmark**
Mount or clone the benchmark first:
```python
!git clone https://github.com/<benchmark-owner>/live-gurbani-captioning-benchmark-v1.git ../live-gurbani-captioning-benchmark-v1
# Or upload audio + GT to Drive and copy in.
```

Then download benchmark audio (one-time, 4 files, ~3 min):
```python
!for id in IZOsmkdmmcg kZhIA8P6xWI kchMJPK9Axs zOtIpxMT9hU; do \
    yt-dlp -x --audio-format wav -o "audio/${id}.wav" "https://youtube.com/watch?v=$id"; \
    ffmpeg -y -i "audio/${id}.wav" -ar 16000 -ac 1 "audio/${id}_16k.wav"; \
    rm "audio/${id}.wav"; \
done
```

Run Path B inference with your fine-tuned adapter:
```python
!python scripts/run_path_b_hmm.py \
    --model-id kdcyberdude/w2v-bert-punjabi --target-lang "" \
    --adapter-dir /content/drive/MyDrive/kirtan_training/lora_v1 \
    --out-dir submissions/pb_kirtan_v1
!python ../live-gurbani-captioning-benchmark-v1/eval.py \
    --pred submissions/pb_kirtan_v1 \
    --gt ../live-gurbani-captioning-benchmark-v1/test/
```

### Colab notes

- **GPU choice matters.** Free T4 ≈ 5x faster than Mac MPS. A100 ≈ 30x faster. For 20+ hour datasets, pay for an A100.
- **Session timeouts.** Free Colab kills idle sessions in 12h. Save checkpoints frequently (set `--save-steps 200`) and resume from Drive.
- **bf16 helps on A100.** Add `--bf16` flag to `finetune_path_b.py` once you confirm it's stable for your run (cuts memory use roughly in half).

## Option 2: RunPod (slightly more setup, no free tier, but cheaper than Colab Pro for serious runs)

RunPod gives you an SSH-able GPU container for ~$0.40-0.80/hr (A100). For a 20-hour-data fine-tune that takes ~3 hours on A100, that's ~$2.

### Setup

1. Sign up at [runpod.io](https://www.runpod.io/). Add credits.
2. Deploy a "PyTorch 2.1" template on an A100 or RTX 4090 pod.
3. SSH in (RunPod gives you a port + key).
4. Inside the pod:
   ```bash
   git clone https://github.com/<your-username>/live-gurbani-captioning.git
   cd live-gurbani-captioning
   pip install -r requirements.txt
   # rsync training_data/ from your local Mac:
   #   rsync -av training_data/ runpod:live-gurbani-captioning/training_data/
   python scripts/finetune_path_b.py --manifest ... --output-dir lora_adapters/kirtan_v1
   # rsync the adapter back when done.
   ```

## Practical tips

### Anti-overfitting checklist
- **Holdout by shabad identity** — split your CSV by `shabad_id`, so no shabad appears in both train and val.
- **Validation manifest** — pass `--eval-manifest path/to/val.json` to `finetune_path_b.py` so it tracks eval loss every 500 steps.
- **Early stopping** — once eval loss plateaus, kill the run. LoRA's low parameter count is itself a regularizer, but it's still possible to overfit on small data.
- **Data augmentation** — for v2+, consider tempo perturbation (±10%) and pitch shift (±2 semitones) at the dataset level. Mimics singer variation.

### Sanity checks before committing to a real run
1. Smoke train for 100 steps on a tiny subset (`--max-steps 100`). Check loss decreases.
2. Run inference with the smoke-trained adapter. Output should be different from the base model.
3. Score on benchmark. Expect noise at this scale, but score should not be dramatically worse than base.

If any of these fail, debug before scaling up. Scaling up bad signal wastes compute.

### Expected results

Published lyrics-alignment and Quran-recitation systems with similar fine-tuning approaches typically report:
- 10-20 hours of in-domain training data → +5-10 points
- 50+ hours of in-domain → +10-15 points
- Plus task-specific architecture tweaks → further +3-5 points

Realistic target for Path B + 20h kirtan fine-tune: **~85-90% blind+live**. With more data and architectural work, **90-95%** is plausible.

## When the fine-tuned adapter is ready

Update `submissions/pb_kirtan_v1/notes.md` with config and score. Commit it. The pipeline is end-to-end from there.

If the result improves over Path A v3.2 (86.5%), this becomes the new canonical Path B. If it doesn't, you've learned that 20h of kirtan data isn't enough to overcome the architectural Path A vs Path B trade-off — and the next move is either more data, a better data curation pipeline, or a different decoding architecture in Path B.

Either outcome is informative.
