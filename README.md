# LAVE2027: Training-Free Visual-State Communication for Active Evidence Acquisition in Long Videos

**LAVE** (*Latent Visual Evidence-Enhanced Planning*, arXiv:2608.07585) is a **100% training-free** visual-state communication framework for long-video tool-use agents on frozen multimodal backbones (`Qwen3.5-9B`).

Instead of discarding precomputed visual representations after a coarse tool call compresses frames into text, LAVE caches post-merger visual states with **zero vision-encoder replay**, scores candidate slots via a **zero-pixel Calibrated-Lift Receiver** under strict content controls, and routes subsequent fine-grained `Focus` / `Skim` calls—and direct zero-pixel latent visual answering—natively inside **vLLM 0.19.0** (`enable_mm_embeds=True`, M-RoPE, zero vLLM source modifications).

---

## Core Design Principles

| Component | Mechanism | Training | Extra Vision Encoder Calls | Native vLLM 0.19.0 Support |
| :--- | :--- | :---: | :---: | :---: |
| **Zero-Replay State Capture** | Read-only forward hook on `model.visual` + `embed_input_ids` captures post-merger embeddings during coarse Tool prefill | None (Frozen 9B) | **0** | `LLM.apply_model` (zero source edits) |
| **Zero-Pixel Calibrated-Lift Receiver** | Scores `lift(i) = [z_Y(e_i) - z_N(e_i)] - [z_Y(0) - z_N(0)]` via `image_embeds` + native 3D M-RoPE + prefix caching | None | **0** *(enforced by forbid hook)* | `enable_mm_embeds=True`, `mamba_cache_mode="align"`, `logprobs_mode="processed_logits"` |
| **Entropy-Guided Multi-Interval Routing** | Normalized Shannon entropy over positive lifts allocates `K ∈ {1, 2, 4}` NMS peaks under strict equal-frame split budget | None | **0** | Full parity verified vs HF |
| **Direct Visual / Latent Readout** | Bypasses the lossy Focus→text summary bottleneck before multiple-choice answering | None | **0** *(latent-embed mode)* | Multi-image `image_embeds` + `image_grid_thw` |

---

## Audited Experimental Results

All evaluations use frozen **`Qwen3.5-9B`** (`temperature=0.0`, `thinking=False`, zero gradient updates, zero LoRA, zero teacher distillation). Every benchmark uses frozen blind preflight manifests (`labels=0`), independent CPU audits, and verified GPU release (`0 MiB`).

### 1. Summary Across Benchmarks

| Benchmark & Split | Cohort | Backbone & Runtime | Primary Metric | `text` Baseline | `clock_split` (Equal Budget) | **`state_split` (LAVE)** | LAVE vs `text` (Paired McNemar) | Receipt |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CG-Bench Dev** (`entropy_routing`) | 51 videos | Qwen3.5-9B (HF) | Focus-1 New Clue Hit (excl. Source) | 15/51 (`29.4%`) | 14/51 (`27.5%`) | **21/51 (`41.2%`)** | **+8 / −2** (`+11.8 pp`; vs clock **+8 / −1**, `p = 0.0391`) | [entropy_score.json](results/cgbench_dev51/entropy_score.json) |
| **CG-Bench Dev + Focus-2** (`zoom_verify`) | 51 videos | Qwen3.5-9B (HF) | Pure-Text QA (`on_policy` / `equal_budget`) | 14/51 (`27.5%`) / 15/51 (`29.4%`) | 15/51 (`29.4%`) / 16/51 (`31.4%`) | **19/51 (`37.3%`)** / **19/51 (`37.3%`)** | **+5 / −0** (`+9.8 pp`, 2-sided `p = 0.0625`, 1-sided `p = 0.0313`) | [zoom_verify.score.json](results/cgbench_dev51/zoom_verify.score.json) |
| **CG-Bench Blind Confirm** *(Clue Loc.)* | **96 unseen** | Qwen3.5-9B (HF) | **Focus-1 New Clue Hit (excl. Source)** | 26/96 (`27.1%`) | 36/96 (`37.5%`) | **41/96 (`42.7%`)** | **+17 / −2** (**`+15.6 pp`, `p = 0.000729`**) | [score.json](results/cgbench_confirm96/score.json) |
| **CG-Bench Blind Confirm** *(Triggered)* | 69 routed | Qwen3.5-9B (HF) | Focus-1 New Clue Hit (excl. Source) | 21/69 (`30.4%`) | 31/69 (`44.9%`) | **36/69 (`52.2%`)** | **`+21.7 pp`**; empty Focus-1 reports cut **31%** (`29 → 20`) | [notes.md](results/cgbench_confirm96/notes.md) |
| **CG-Bench Blind Confirm** *(Pure-Text QA)* | **96 unseen** | Qwen3.5-9B (HF) | Native Pure-Text QA (F1 / F1+2) | 34/96 (`35.4%`) / 33/96 (`34.4%`) | 33/96 (`34.4%`) / 34/96 (`35.4%`) | 33/96 (`34.4%`) / 32/96 (`33.3%`) | +0/−1 (`p = 1.0`) / +1/−2 (`p = 1.0`) *(honest null; Focus→text bottleneck)* | [score.json](results/cgbench_confirm96/score.json) |
| **LongVideoBench-150** *(All 17 Cats)* | **150 (146 vids)** | Qwen3.5-9B (**vLLM 0.19.0**) | **Direct Visual QA (Focus-1+2)** | 83/150 (`55.3%`) | 85/150 (`56.7%`) | **87/150 (`58.0%`)** | **+5 / −1** (**`+2.67 pp`**, `p = 0.219`; **0/17 category regressions**) | [score.json](results/longvideobench_150/score.json) |
| **LongVideoBench-150** *(Routed Subset)* | 108 routed | Qwen3.5-9B (**vLLM 0.19.0**) | Direct Visual QA (Focus-1+2) | 67/108 (`62.0%`) | 69/108 (`63.9%`) | **71/108 (`65.7%`)** | **+5 / −1** (**`+3.70 pp`**; vs pure-text `text` `53.3%` → **`+4.67 pp` total**) | [notes.md](results/longvideobench_150/notes.md) |
| **Video-Holmes Pilot** *(Multi-Step VideoSeek+LAVE)* | 14 (14 vids, 7×2) | Qwen3.5-9B (**vLLM 0.19.0**) | Multi-Step (`max_steps=6`) Acc / Frames / Calls | `8/14 (57.1%)` (`72.0f`, `3.50` calls, `4.36` st) | — | **`8/14 (57.1%)`** (**`59.4f` −17.5%**, **`2.71` calls −22.6%**, `3.71` st) | Same accuracy (`+0/−0`) with **12.6 fewer frames** & **−22.6% visual calls** (`24/24` controls) | [score_pilot14.json](results/video_holmes/score_pilot14.json) |
| **Video-Holmes Eval** *(Multi-Step VideoSeek+LAVE)* | **140 (140 vids, 7×20)** | Qwen3.5-9B (**vLLM 0.19.0**) | 6-Arm Multi-Step (`step4` + `step6` + `top-8 latent verify`) | *Running on A40 GPU 1* | — | *In progress* | *Disjoint from pilot14; single-video self-controls + 0-pixel latent verify* | [results/video_holmes/](results/video_holmes/) |

---

### 2. CG-Bench 96 Blind Independent Confirmation Details (`results/cgbench_confirm96/`)

Strict blind protocol (`labels=0`, `clues=0`, `training=0`, `invalid_qa_cells=0`, peak `19.95 GiB`, GPU 0 released to `0 MiB`). Strict 3-control pass rate: **`69 / 83` (`83.1%`)** (`K=4`: 65, `K=2`: 4, `K=0`: 27).

| Endpoint (`N = 96` unseen long videos) | `text` | `clock_split` | **`state_split` (LAVE)** | `state` vs `text` (`p`) | `state` vs `clock` (`p`) | Avg Turns / Frames |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Focus-1 New Clue Hit (excl. Source)** | 26 (`27.1%`) | 36 (`37.5%`) | **41 (`42.7%`)** | **`+17 / −2` (`p = 0.000729`)** | `+11 / −6` (`p = 0.332`) | `1.91` / `60.49` |
| **Focus-2 Clue Hit (`on_policy`)** | 11 (`11.5%`) | 13 (`13.5%`) | **16 (`16.7%`)** | `+9 / −4` | `+8 / −5` | `2.35–2.44` / `67.7–69.0` |
| **Focus-2 Clue Hit (`equal_budget`)** | 13 (`13.5%`) | 19 (`19.8%`) | **20 (`20.8%`)** | `+10 / −3` | `+7 / −6` | `2.50` / `69.99` |
| Native Pure-Text QA (Focus-1) | 34 (`35.4%`) | 33 (`34.4%`) | 33 (`34.4%`) | `+0 / −1` (`p = 1.0`) | `+0 / −0` (`p = 1.0`) | `1.91` / `60.49` |
| Native Pure-Text QA (Focus-1+2) | 33 (`34.4%`) | 34 (`35.4%`) | 32 (`33.3%`) | `+1 / −2` (`p = 1.0`) | `+1 / −3` (`p = 0.625`) | `2.50` / `69.99` |

**Autopsy of the Clue→QA Bottleneck**: Across the **69** confirmed cases where `state_split` inspected different Focus-1 frames than `text`, **`67 / 69` (`97.1%`)** chose the exact same pure-text QA-1 option (`64 / 69` in QA-2), driven by native Focus prompt query-echoing and strong pure-text QA anchoring on the two initial Source reports.

---

### 3. Focus-2 Readout Ablation & Direct Visual QA Ceiling (`results/cgbench_dev51/focus2_synth.score.json`)

Tested on the exact same 16 sealed Focus-2 JPEGs (`N = 51` dev videos, `peak 19.29 GiB`, zero touch on Confirm-96):

| Readout Variant (`N = 51` CG-Bench Dev) | `on_policy` (`state / text / clock`) | `equal_budget` (`state / text / clock`) | Key Observation |
| :--- | :---: | :---: | :--- |
| Native Focus-2 → Pure-Text QA | **19 / 14 / 15** | **19 / 15 / 16** | Query-echoing loop on counting cases |
| Cross-Frame Synth Focus-2 → Native QA (`synth_qa_native`) | **19 / 16 / 16** (`+3/−0` vs `text`) | **19 / 17 / 19** | Rescues counting cases (`case_17`, `case_48`), adds 2 hallucination flips |
| Cross-Frame Synth Focus-2 → Priority QA (`synth_qa_priority`) | 17 / 15 / 16 | — | Over-trusting free-form Focus-2 text amplifies hallucinations |
| **Direct Visual QA Ceiling** (16 raw frames + `A–H`) | **18 / 16 / 14** | **18 / 18 / 16** | Frozen 9B fine-counting ceiling on 8-choice CG-Bench (~`35–37%`) |

---

### 4. LongVideoBench-150 Category-Balanced Evaluation (`results/longvideobench_150/`)

Frozen blind manifest (`150` questions across `146` unique long videos `>= 180s`: `101` in `duration_group=3600`, `49` in `duration_group=600`; all **17** LVB categories balanced `8–9` per category). Dual-GPU A40 (`vLLM 0.19.0`, `gpu_memory_utilization=0.90`, `mm_processor_cache_gb=0`).

- **Cross-Dataset Control Stability**: **`108 / 130` (`83.1%`)** passed all 3 strict controls (`K=4`: 101, `K=2`: 5, `K=1`: 2) — identical to CG-Bench Confirm-96 (`83.1%`).
- **Turns & Frames**: Focus-1 avg **`1.91` turns / `60.71` frames**; Focus-1+2 avg **`2.37` turns / `67.96` frames**.
- **vLLM Runtime**: 3-arm parallel total **`70.28 s`/case** (`~2.0×` faster than HF despite 6 extra visual QA calls); **LAVE Receiver (33 slots + 3 controls, 0 vision calls) = `5.07 s`/case**.

| Endpoint (`N = 150` LVB) | `text` | **`state_split` (LAVE)** | `clock_split` | `state` vs `text` (`p`) | `state` vs `clock` (`p`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`direct_visual_qa_f2`** (`op` / `eq` identical) | 83 (`55.3%`) | **87 (`58.0%`)** | 85 (`56.7%`) | **`+5 / −1` (`+2.67 pp`, `p = 0.219`)** | `+4 / −2` (`+1.33 pp`, `p = 0.688`) |
| `direct_visual_qa_f1` | 84 (`56.0%`) | **85 (`56.7%`)** | 84 (`56.0%`) | `+3 / −2` (`+0.67 pp`, `p = 1.000`) | `+3 / −2` (`p = 1.000`) |
| Pure-Text `focus1_native` | 80 (`53.3%`) | 81 (`54.0%`) | 82 (`54.7%`) | `+1 / −0` (`+0.67 pp`, `p = 1.000`) | `+0 / −1` (`p = 1.000`) |
| Pure-Text `focus12` (`op` / `eq` identical) | 80 (`53.3%`) | 81 (`54.0%`) | 81 (`54.0%`) | `+1 / −0` (`+0.67 pp`, `p = 1.000`) | `+1 / −1` (`p = 1.000`) |

**Breakdown by Duration Group (`direct_visual_qa_f2`)**:

| Duration Group | `n` | `text` | **`state_split` (LAVE)** | `clock_split` | `state` vs `text` | `state` vs `clock` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `600` (3–10 min) | 49 | 29 (`59.2%`) | **30 (`61.2%`)** | 28 (`57.1%`) | `+1 / −0` (`+2.0 pp`) | **`+2 / −0` (`+4.1 pp`)** |
| `3600` (10–60 min) | 101 | 54 (`53.5%`) | **57 (`56.4%`)** | 57 (`56.4%`) | **`+4 / −1` (`+3.0 pp`)** | `+2 / −2` (`0.0 pp`) |
| **All 17 Categories (`state` vs `text`)** | **150** | — | **4 wins / 13 ties / 0 losses** | — | **Zero regressions across all 17 categories** (`E2O`, `O2E`, `S2O`, `T2E` gain) | — |

---

### 5. Native vLLM 0.19.0 Parity & Speedup Benchmark (`results/vllm_parity/bench_vllm_b33.json`)

4-case A40 benchmark (`case_16 / 06 / 20 / 38`, frozen `Qwen3.5-9B`, zero vLLM source patches, `forbid_vision_hook` enforcing `0` vision calls in Receiver):

| Stage | HF Transformers (A40) | **Native vLLM 0.19.0 (A40)** | Speedup | Parity Verification |
| :--- | :---: | :---: | :---: | :--- |
| **LAVE Receiver (33 slots + 3 controls)** | `14.60 s` | **`4.36 s`** (`b=1`: `4.41 s`) | **`3.35×`** | `4/4` grids exact `[1,24,42]`; `4/4` controls pass (`K=4`); `0` vision calls |
| `source_shared` (32 frames + hook capture) | `29.99 s` | **`20.78 s`** | `1.44×` | `zero_margin_diff = 0.000–0.125`; `case_16` exact 4-gap set `{0,4,6,8}` |
| `Focus-1` (32 frames) | `8.82 s` | **`7.57 s`** | `1.17×` | `case_06` exact Top-3 `[2,7,27]` |

---

## Repository Structure

```text
LAVE2027/
├── README.md                        # Overview, result tables, and reproduction guide
├── missing_fact_routing_20261124/   # Calibrated-lift receiver, control gating, native runtime
├── entropy_routing_20261125/        # Entropy-guided NMS multi-interval routing + Focus-2 verification
├── confirm96_20261126/              # CG-Bench 96 unseen blind confirmation runner, auditor, scorer
├── focus2_synth_20261126/           # Focus-2 synthesis & Direct Visual QA ceiling ablation
├── vllm_lave_20261126/              # Native vLLM 0.19.0 zero-replay LAVE engine (0 source patches)
├── lvb_vllm_20261126/               # LongVideoBench-150 dual-GPU vLLM runner & single-unseal scorer
├── videoseek_lave_20261126/         # Multi-step VideoSeek + LAVE (single-video self-controls + 0-pixel latent QA)
├── results/
│   ├── cgbench_dev51/               # Dev-51 release/score JSONs, Focus-2 synthesis & batched bench
│   ├── cgbench_confirm96/           # Confirm-96 sealed protocol, release.json, score.json, notes
│   ├── vllm_parity/                 # vLLM 0.19.0 parity & 3.35x speedup benchmark receipt
│   ├── longvideobench_150/          # LVB-150 frozen preflight, shard terminals, score.json, notes
│   └── video_holmes/                # Video-Holmes multi-step VideoSeek+LAVE receipts
└── docs/
    └── RESEARCH_MEMORY.md           # Full chronological research & audit trail
```

---

## Quick Verification & Reproduction

### CPU Unit Tests (No GPU Required)
```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s vllm_lave_20261126 -p "test_*.py" -v
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s lvb_vllm_20261126 -p "test_*.py" -v
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s videoseek_lave_20261126 -p "test_*.py" -v
```

### Multi-Step VideoSeek + LAVE on vLLM 0.19.0 (Single GPU)
```bash
# 1. Freeze blind Video-Holmes manifests (labels=0)
CUDA_VISIBLE_DEVICES="" python -m videoseek_lave_20261126.run_holmes preflight pilot14
CUDA_VISIBLE_DEVICES="" python -m videoseek_lave_20261126.run_holmes preflight eval140

# 2. Run blind evaluation on a single GPU
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=1 \
  python -B -u -m videoseek_lave_20261126.run_holmes run eval140 <PREFLIGHT_PIN> 1 0.90 6

# 3. Single-unseal CPU scoring (verifies GPU release)
CUDA_VISIBLE_DEVICES="" python -B -u -m videoseek_lave_20261126.score_holmes eval140
```
