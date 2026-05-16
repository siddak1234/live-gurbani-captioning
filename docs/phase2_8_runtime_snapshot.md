# Phase 2.8 runtime snapshot — current ASR state

Captured: 2026-05-16, immediately after Phase 2.7 failed.

Purpose: preserve the runtime state that produced:

- `submissions/v3_2_repro_current`: **73.5%**
- `submissions/v5b_idlock_runtime`: **75.6%**

This is the starting point for Phase 2.8 Workstream A.

## Package Versions

| Package | Version |
|---|---|
| Python | 3.12.13 |
| faster-whisper | 1.2.1 |
| CTranslate2 | 4.7.1 |
| tokenizers | 0.20.3 |
| transformers | 4.46.3 |
| torch | 2.5.0 |
| huggingface-hub | 0.36.2 |
| platform | macOS-26.3.1-arm64-arm-64bit |

## Model Snapshot

faster-whisper `medium` resolved to:

```text
~/.cache/huggingface/hub/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f
```

## ASR Cache Checksums

Current faster-whisper-medium cache:

| Cache file | sha256 |
|---|---|
| `asr_cache/IZOsmkdmmcg_16k__medium__pa.json` | `dcb29deabe8cba1a611b8e1e784e0a56db72de9291b8e11314e43fefb541e85f` |
| `asr_cache/kZhIA8P6xWI_16k__medium__pa.json` | `e50212bcc8c4724f19eae460c6b548fbd8a4953b525980040fa84245eeafb027` |
| `asr_cache/kchMJPK9Axs_16k__medium__pa.json` | `cf54b1bea1d851f341e1938a55fb958404348ae4e8ac1e311385c646d9ede4e7` |
| `asr_cache/zOtIpxMT9hU_16k__medium__pa.json` | `1d582f286a239bff1b0a251a07cbdd297a06dfd9af55963fedd065fbbbf3ecb9` |

Current v5b HF adapter cache:

| Cache file | sha256 |
|---|---|
| `asr_cache/IZOsmkdmmcg_16k__hf-surindersinghssj_surt-small-v3_w10_lora-v5b_mac_diverse__pa.json` | `7b7257bf9b9820e7ed59feec604d23cbaf3f9479eb907b8e5a4837a11b8c9f62` |
| `asr_cache/kZhIA8P6xWI_16k__hf-surindersinghssj_surt-small-v3_w10_lora-v5b_mac_diverse__pa.json` | `1f23c41932ab76614a0874178980559dd80f94bb6749bef948275922949f765b` |
| `asr_cache/kchMJPK9Axs_16k__hf-surindersinghssj_surt-small-v3_w10_lora-v5b_mac_diverse__pa.json` | `08cd171e35af88d98f505635b71bc2e09101c2c95e8ea9ddd9f9853ea80741cd` |
| `asr_cache/zOtIpxMT9hU_16k__hf-surindersinghssj_surt-small-v3_w10_lora-v5b_mac_diverse__pa.json` | `328855736ea2af8358bbf646aca21755df60c932f8f0d1d97223b3289e99a319` |

## First 30s Failure Evidence

The current `chunk_vote` blind-ID failure on `kZhIA8P6xWI` comes from poor first-window transcription:

### `kZhIA8P6xWI`

```text
gt=1821 pred=4377 top=213.4 runner_up=1341:0.0
. . . . . . . . . . . . cloaked with gold.. भिशरे मेरे प्रान उंडार जीतम क्यों भिशरे मेरे प्रान उंडार
```

### `kZhIA8P6xWI_cold33`

```text
gt=1821 pred=4377 top=171.0 runner_up=1821:79.6
ताम का आंप ना जाए लक्ना आउत जाए गुभार वीतम क्यों बिस रहे मेरे प्रान अगाँ रव सास दीपक जान कत्र भुण एका जोत मुरार
```

### `kZhIA8P6xWI_cold66`

```text
gt=1821 pred=1821 top=139.4 runner_up=4377:85.5
अंतर जोत सबक तुम जाकें सबकूर चग्रन बेरा वीतम क्यों बिस रहे मेरे प्रान अगाँ शुर्नार नात बेहांत अजोणी साचे महली पारा
```

The model is producing noisy Devanagari/Hindi-like text rather than stable Punjabi transliteration in the failure windows. That makes line-similarity based shabad-ID fragile no matter which simple aggregator is used.

## Immediate Next Probe

Before bigger LoRA training, test timestamp/transcription variants against this exact snapshot:

1. faster-whisper `vad_filter=False`;
2. faster-whisper `word_timestamps=True`;
3. both together;
4. hybrid: v5b/surt text with faster-whisper timing.

If none recover the 86-90% range, move to full-shabad forced alignment.
