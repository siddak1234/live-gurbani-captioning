# Phase 2.10 silver source-row audit

Parquet source: `/Users/sbhatia/.cache/huggingface/hub/datasets--surindersinghssj--gurbani-kirtan-yt-captions-300h-canonical/snapshots/e805818eefbf4e4ae70c389c572114fa1b8fbf5f/data`
Weak threshold: best(base, v5b) WRatio < `90.0`

## Summary

| Assessment | Rows |
| --- | --- |
| silver-label-risk: heavy canonical fixes | 1 |
| silver-label-risk: prediction matches raw caption better than canonical final | 10 |

## Weak-row source metadata

| idx | clip | video | decision | margin | raw->pred | final->pred | assessment | raw text | canonical final | best pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 535 | 2d_Wy2Vb6n4_clip_00006 | 2d_Wy2Vb6n4 | replaced | 0.728 | 85.7 | 84.7 | silver-label-risk: heavy canonical fixes | ਬੀਆ >> ਸੰਤਨ ਬਿਨ ਅਵਰ ਨ ਦਾਤਾ | ਥੀਆ ਸੰਤਨ ਬਿਨੁ ਅਵਰੁ ਨ ਦਾਤਾ | ਪੀਆ ਸੰਤਨ ਬਿਨੁ ਅਗਨਿ ਨ ਦਾਤਾ |
| 3233 | PYUPZn6wiR8_clip_00000 | PYUPZn6wiR8 | review | 0.434 | 100.0 | 80.0 | silver-label-risk: prediction matches raw caption better than canonical final | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ | ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ |
| 3234 | PYUPZn6wiR8_clip_00001 | PYUPZn6wiR8 | review | 0.702 | 88.9 | 76.2 | silver-label-risk: prediction matches raw caption better than canonical final | ਸਦਾ ਜਾਗੇ >> ਰਾਮ ਪਿਆਰੇ | ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ | ਸਦਾ ਕੇ ਰਾਮ ਪਿਆਰੇ |
| 3235 | PYUPZn6wiR8_clip_00002 | PYUPZn6wiR8 | review | 0.684 | 100.0 | 80.0 | silver-label-risk: prediction matches raw caption better than canonical final | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ | ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ |
| 3236 | PYUPZn6wiR8_clip_00019 | PYUPZn6wiR8 | review | 0.417 | 100.0 | 80.0 | silver-label-risk: prediction matches raw caption better than canonical final | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ | ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ |
| 3237 | PYUPZn6wiR8_clip_00021 | PYUPZn6wiR8 | review | 0.679 | 91.7 | 80.0 | silver-label-risk: prediction matches raw caption better than canonical final | ਸਦਾ ਜਾਗੈ >> ਰਾਮ ਪਿਆਰੇ | ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ | ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ |
| 4542 | iQAbsSM5FO8_clip_00037 | iQAbsSM5FO8 | review | 0.082 | 83.3 | 58.8 | silver-label-risk: prediction matches raw caption better than canonical final | ਫਲ ਦੈ ਲਾ ਭੁਖ | ਮਃ ਮਾਰੂ ਭੁਖ | ਫਲ ਦੇ ਲਾਭੁ ਭੁਖ |
| 4543 | iQAbsSM5FO8_clip_00048 | iQAbsSM5FO8 | review | 0.271 | 80.0 | 68.6 | silver-label-risk: prediction matches raw caption better than canonical final | ਆਪ ਸੰਦੇ ਦੁਖ | ਆਪੇ ਸਿਰਿ ਲੇਖੈ | ਆਪੇ ਸਖਲੇ ਦੁਖ |
| 4544 | iQAbsSM5FO8_clip_00055 | iQAbsSM5FO8 | replaced | 0.069 | 100.0 | 66.7 | silver-label-risk: prediction matches raw caption better than canonical final | ਮਨਿ ਪਰਚਾਇਆ | ਧਨਿ ਰੁਚ ਇਆ | ਮਨਿ ਪਰਚਾਇਆ |
| 4545 | iQAbsSM5FO8_clip_00056 | iQAbsSM5FO8 | replaced | 0.125 | 87.0 | 75.0 | silver-label-risk: prediction matches raw caption better than canonical final | >> ਮਨ ਪਰਚਾਇਆ | ਧਨਿ ਰੁਚ ਇਆ | ਧਨ ਪਰਚਾਇਆ |
| 4546 | iQAbsSM5FO8_clip_00059 | iQAbsSM5FO8 | replaced | 0.091 | 100.0 | 66.7 | silver-label-risk: prediction matches raw caption better than canonical final | ਮਨਿ ਪਰਚਾਇਆ | ਧਨਿ ਰੁਚ ਇਆ | ਮਨਿ ਪਰਚਾਇਆ |

## Read

- Most weak rows are silver-label risks, not evidence that the model cannot hear the line.
- The model often matches the raw caption / audible phrase better than the canonical replacement.
- This confirms the current protocol: use silver for diagnostics, but keep gold OOS for promotion.
- Next experiment should not be broad adapter scaling. Inspect weak audio/labels or improve runtime alignment.
