# live-gurbani-captioning — minimal-run Makefile.
#
# One target per workflow step. Each wraps a single script; defaults live in
# configs/. To customize, edit the variables at the top of this file or pass
# overrides on the command line:
#     make train TRAIN_OUT=lora_adapters/my_run
#
# Fresh-clone training Mac, zero config:
#     make start
# That runs: doctor-train → install → data → train. No ffmpeg, no benchmark
# repo, no smoke test required. The training machine pulls HuggingFace data
# and fine-tunes; everything else is dev-machine work.
#
# Fresh-clone dev Mac (benchmark scoring, iOS build):
#     make start-dev
# That adds doctor-dev (checks ffmpeg + benchmark repo) and fetch-audio +
# corpus + smoke before training.
#
# Each step is idempotent — safe to interrupt and re-run.

# -----------------------------------------------------------------------------
# Tunable defaults
# -----------------------------------------------------------------------------

PYTHON         ?= python3
REQUIREMENTS   ?= requirements-mac.txt
DATA_DIR       ?= training_data/kirtan_v1
DATA_SAMPLES   ?= 200
DATA_MIN_SCORE ?= 0.8
DATA_MAX_SCAN  ?= 5000
DATA_SHARD     ?= 0
DATA_SHARDS    ?=
DATA_MIN_UNIQUE_VIDEOS  ?= 0
DATA_MIN_UNIQUE_SHABADS ?= 0
DATA_FORCE     ?= 0
SMOKE_OUT      ?= /tmp/lora_smoke
TRAIN_OUT      ?= lora_adapters/surt_mac_v1
TRAIN_CFG      ?= configs/training/surt_lora_mac.yaml
INFER_CFG      ?= configs/inference/v3_2.yaml
EVAL_OUT       ?= submissions/v5_surt_mac_v1
COREML_CFG     ?= configs/export/coreml_ane.yaml
COREML_OUT     ?= ios/Sources/GurbaniCaptioning/Resources
BENCHMARK_DIR  ?= ../live-gurbani-captioning-benchmark-v1
HF_WINDOW_SECONDS ?= 10
OOS_SHABAD_ID  ?=
OOS_URL        ?=
OOS_CLIP       ?=
DATA_SHARDS_ARG := $(if $(DATA_SHARDS),--shards $(DATA_SHARDS),--shard $(DATA_SHARD))

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help (default target).
	@printf "live-gurbani-captioning — minimal-run targets\n\n"
	@grep -hE '^[a-z][a-z0-9_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F ':.*## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@printf "\nTraining machine, zero config:\n  make start\n"
	@printf "\nDev machine (scoring + iOS):\n  make start-dev\n"

# -----------------------------------------------------------------------------
# Doctor — system-dep checks. Two flavors: training-only vs full dev.
# -----------------------------------------------------------------------------

.PHONY: doctor-train
doctor-train: ## Check Python >=3.10 (training machine only — no ffmpeg, no benchmark).
	@printf "checking training-machine deps...\n"
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "  ✗ $(PYTHON) not found. Install Python 3.10+: brew install python@3.12"; exit 1; }
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3,10), f'need 3.10+, got {sys.version_info[:2]}'" \
		|| { echo "  Hint: brew install python@3.12 && export PATH=\"/opt/homebrew/bin:\$$PATH\""; exit 1; }
	@printf "  ✓ $(PYTHON) $$($(PYTHON) --version | cut -d' ' -f2)\n"

.PHONY: doctor-dev
doctor-dev: doctor-train ## Check Python + ffmpeg + paired benchmark repo (dev machine).
	@command -v ffmpeg >/dev/null 2>&1 || { \
		echo "  ✗ ffmpeg not found. Install: brew install ffmpeg"; exit 1; }
	@printf "  ✓ ffmpeg $$(ffmpeg -version 2>&1 | head -1 | awk '{print $$3}')\n"
	@test -d $(BENCHMARK_DIR)/test || { \
		echo "  ✗ paired benchmark repo not found at $(BENCHMARK_DIR)"; \
		echo "    Clone it alongside this repo:"; \
		echo "    cd .. && git clone <benchmark-repo-url> live-gurbani-captioning-benchmark-v1"; \
		exit 1; }
	@printf "  ✓ benchmark repo at $(BENCHMARK_DIR)\n"

# Back-compat: `make doctor` still works, alias to doctor-dev (the more thorough check).
.PHONY: doctor
doctor: doctor-dev ## Alias for doctor-dev.

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: install
install: ## Install Mac dependencies (requirements-mac.txt).
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

.PHONY: corpus
corpus: ## Build the BaniDB corpus cache + iOS bundle JSON.
	$(PYTHON) scripts/build_corpus.py
	$(PYTHON) scripts/build_ios_corpus.py

.PHONY: corpus-oos
corpus-oos: ## Cache one OOS shabad: make corpus-oos OOS_SHABAD_ID=5621
	@test -n "$(OOS_SHABAD_ID)" || { \
		echo "Usage: make corpus-oos OOS_SHABAD_ID=5621"; exit 1; }
	$(PYTHON) scripts/build_corpus.py --shabad-id $(OOS_SHABAD_ID)
	$(PYTHON) scripts/build_ios_corpus.py

.PHONY: fetch-audio
fetch-audio: ## Download benchmark audio via yt-dlp (idempotent — skips existing files).
	$(PYTHON) scripts/fetch_audio.py

.PHONY: fetch-oos-audio
fetch-oos-audio: ## Download one OOS URL: make fetch-oos-audio OOS_URL='case_001=https://...' OOS_CLIP='case_001=30-210'
	@test -n "$(OOS_URL)" || { \
		echo "Usage: make fetch-oos-audio OOS_URL='case_001=https://...'"; exit 1; }
	$(PYTHON) scripts/fetch_audio.py \
		--audio-dir eval_data/oos_v1/audio \
		--url "$(OOS_URL)" \
		$(if $(OOS_CLIP),--clip "$(OOS_CLIP)",)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

.PHONY: data
data: ## Pull a labeled kirtan slice from HuggingFace (idempotent — skips if manifest exists).
	@if [ "$(DATA_FORCE)" != "1" ] && [ -f "$(DATA_DIR)/manifest.json" ]; then \
		echo "skip: $(DATA_DIR)/manifest.json already present (delete it or set DATA_FORCE=1 to re-pull)"; \
	else \
		$(PYTHON) scripts/pull_dataset.py kirtan \
			--out-dir $(DATA_DIR) \
			--num-samples $(DATA_SAMPLES) \
			--min-score $(DATA_MIN_SCORE) \
			--max-scan $(DATA_MAX_SCAN) \
			$(DATA_SHARDS_ARG) \
			--min-unique-videos $(DATA_MIN_UNIQUE_VIDEOS) \
			--min-unique-shabads $(DATA_MIN_UNIQUE_SHABADS); \
	fi

.PHONY: data-v5b
data-v5b: ## Phase 2.5 diagnostic pull: larger/diverse held-out kirtan slice.
	$(MAKE) data \
		DATA_DIR=training_data/v5b_mac_diverse \
		DATA_SAMPLES=1000 \
		DATA_MIN_SCORE=0.85 \
		DATA_SHARDS=0-9 \
		DATA_MAX_SCAN=20000 \
		DATA_MIN_UNIQUE_VIDEOS=20 \
		DATA_MIN_UNIQUE_SHABADS=100 \
		DATA_FORCE=1

.PHONY: smoke-manifest
smoke-manifest: ## Build the 4-snippet smoke manifest (validates pipeline only).
	$(PYTHON) scripts/build_smoke_manifest.py

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

.PHONY: smoke
smoke: fetch-audio smoke-manifest ## 20-step fine-tune to validate the entire training pipeline.
	$(PYTHON) scripts/finetune_path_b.py \
		--config $(TRAIN_CFG) \
		--manifest training_data/smoke/manifest.json \
		--output-dir $(SMOKE_OUT) \
		--max-steps 20 --batch-size 1 --report-to none

.PHONY: train
train: data ## Full LoRA fine-tune of surt-small-v3 on $(DATA_DIR). Auto-pulls data if missing.
	$(PYTHON) scripts/finetune_path_b.py \
		--config $(TRAIN_CFG) \
		--manifest $(DATA_DIR)/manifest.json \
		--output-dir $(TRAIN_OUT)

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

.PHONY: eval
eval: ## Score $(TRAIN_OUT) adapter against the paired benchmark.
	HF_WINDOW_SECONDS=$(HF_WINDOW_SECONDS) $(PYTHON) scripts/run_path_a.py \
		--backend huggingface_whisper \
		--model surindersinghssj/surt-small-v3 \
		--adapter-dir $(TRAIN_OUT) \
		--blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
		--stay-bias 6 --blind --blind-aggregate chunk_vote \
		--blind-lookback 30 --live --tentative-emit \
		--out-dir $(EVAL_OUT)
	$(PYTHON) $(BENCHMARK_DIR)/eval.py \
		--pred $(EVAL_OUT)/ \
		--gt   $(BENCHMARK_DIR)/test/

.PHONY: eval-baseline
eval-baseline: ## Re-score Path A v3.2 baseline (regression check; no fine-tune).
	HF_WINDOW_SECONDS=$(HF_WINDOW_SECONDS) $(PYTHON) scripts/run_path_a.py \
		--blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
		--stay-bias 6 --blind --blind-aggregate chunk_vote \
		--blind-lookback 30 --live --tentative-emit \
		--out-dir submissions/v3_2_replay
	$(PYTHON) $(BENCHMARK_DIR)/eval.py \
		--pred submissions/v3_2_replay/ \
		--gt   $(BENCHMARK_DIR)/test/

.PHONY: eval-oos
eval-oos: ## Out-of-set eval — the honest accuracy number.
	$(PYTHON) scripts/eval_oos.py \
		--data-dir eval_data/oos_v1 \
		--pred-dir submissions/oos_v1_$(notdir $(TRAIN_OUT)) \
		--engine-config $(INFER_CFG)

.PHONY: eval-oos-loop-align
eval-oos-loop-align: ## OOS eval for current best runtime: Phase 2.9 loop-align ID-lock.
	HF_WINDOW_SECONDS=10 $(PYTHON) scripts/run_idlock_path.py \
		--gt-dir eval_data/oos_v1/test \
		--audio-dir eval_data/oos_v1/audio \
		--out-dir submissions/oos_v1_phase2_9_loop_align \
		--post-adapter-dir lora_adapters/v5b_mac_diverse \
		--post-context buffered \
		--merge-policy retro-buffered \
		--pre-word-timestamps \
		--smoother loop_align
	$(PYTHON) $(BENCHMARK_DIR)/eval.py \
		--pred submissions/oos_v1_phase2_9_loop_align \
		--gt   eval_data/oos_v1/test

# -----------------------------------------------------------------------------
# iOS export
# -----------------------------------------------------------------------------

.PHONY: ios-export
ios-export: ## Merge LoRA into base + export .mlpackage to ios bundle.
	$(PYTHON) scripts/export_coreml.py \
		--config $(COREML_CFG) \
		--adapter-dir $(TRAIN_OUT) \
		--output-dir $(COREML_OUT)

.PHONY: ios-benchmark
ios-benchmark: ## Measure Core ML latency on macOS (proxy for iPhone perf).
	$(PYTHON) scripts/benchmark_ane_latency.py \
		--mlpackage $(COREML_OUT)/surt-small-v3-kirtan.mlpackage \
		--test-clip audio/IZOsmkdmmcg_16k.wav \
		--warmup 2 --iters 5

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

.PHONY: test
test: ## Run the unit-test suite (stdlib unittest; no torch/transformers needed).
	$(PYTHON) -m unittest discover -s tests -v

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and smoke artifacts. Does NOT touch submissions/ or lora_adapters/.
	rm -rf $(SMOKE_OUT) /tmp/v3_2_replay
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

.PHONY: clean-caches
clean-caches: ## Wipe ASR/corpus caches (forces re-fetch on next run).
	rm -rf asr_cache asr_cache_vocals mms_cache corpus_cache

# -----------------------------------------------------------------------------
# Aggregate targets
# -----------------------------------------------------------------------------

.PHONY: bootstrap
bootstrap: install corpus smoke ## install + corpus + smoke. Dev-machine validation only.

.PHONY: start
start: doctor-train install train ## TRAINING MACHINE: doctor + install + train. No benchmark deps.
	@echo "✓ training complete. adapter at $(TRAIN_OUT)"
	@echo "  send the adapter dir back to the dev machine for scoring + iOS export."

.PHONY: start-dev
start-dev: doctor-dev install fetch-audio corpus data smoke train ## DEV MACHINE: full chain incl. smoke + benchmark.
	@echo "✓ training complete. adapter at $(TRAIN_OUT)"
	@echo "  next: make eval        # benchmark score"
	@echo "        make eval-oos    # honest accuracy on held-out shabads"
	@echo "        make eval-oos-loop-align  # current Phase 2.9 runtime on OOS"
	@echo "        make ios-export  # convert to Core ML for the iOS app"
