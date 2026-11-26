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
| **Video-Holmes Eval** *(Multi-Step VideoSeek+LAVE)* | **140 (140 vids, 7×20)** | Qwen3.5-9B (**vLLM 0.19.0**) | **Multi-Step (`max_steps=6`) Overall / Rerouted** | `40/140 (28.6%)` (`77.7f`, `3.86` calls) | `48/140 (34.3%)` (`77.9f`, `3.87` calls) | **`50/140 (35.7%)`** / `lave_full 48 (34.3%)` (**`76.0f`**, **`3.75` calls**) | **+18 / −8** (**`+7.14 pp`**, 2-sided `p = 0.0755`, 1-sided `p = 0.0378`; beats `VideoSeek-Image` `28.6%` by **`+7.14 pp`**) | [score_eval140.json](results/video_holmes/score_eval140.json) |
| **Video-MME 216** *(72S / 72M / 72L, 12 tasks)* | **216 unique vids** | Qwen3.5-9B (**vLLM 0.19.0**) | **Multi-Step (`max_steps=6`) Overall (S / M / L)** | `125/216 (57.9%)` (`77.8 / 55.6 / 40.3`) | — | **`lave_full 138 (63.9%)`** / **`v_lat 141 (65.3%)`** / `l_rt 131 (60.6%)` (**`64.7f`**, **`3.04` calls −8.7%**) | `lave_full` **+26/−13 (`+6.02 pp`, `p = 0.053`)**; `v_lat` **+23/−7 (`+7.41 pp`, `p = 0.0052`)**; **0/3 duration regressions** | [score_mme.json](results/videomme_216/score_mme.json) |
| **LongVideoBench-150** *(Multi-Step, 17 cats)* | **150 (146 vids)** | Qwen3.5-9B (**vLLM 0.19.0**) | **Multi-Step (`max_steps=6`) Overall / 1h (`dg=3600`, `n=101`)** | `85/150 (56.7%)` / `53.5%` (`96.4f`, `5.03` calls) | — | **`l_rt 87/150 (58.0%)`** / **`59.4% (+5.94 pp)`** (**`91.6f`**, **`4.73` calls −6.0%**) | **`+5.94 pp`** on ultra-long 1-hour videos with **−6.0% visual calls** & **−4.8 frames** | [score_lvb150_multistep.json](results/lvb150_multistep/score_lvb150_multistep.json) |
| **Video-Holmes Cross-Scale** | **140 unique vids** | **`Qwen3.5-4B`** (**vLLM 0.19.0**) | Multi-Step (`max_steps=6`) Acc | `35/140 (25.0%)` (`72.2f`, `3.51` calls) | — | `l_full 39 (27.9%)` / **`v_lat 52/140 (37.1%)`** (**`69.5f`**, **`3.34` calls**) | `v_lat` **+22 / −5 (`+12.14 pp`, `p = 0.00151`)** — beats frozen 9B text VideoSeek (`28.6%`) by **`+8.57 pp`**! | [score_qwen35_4b.json](results/video_holmes/score_qwen35_4b.json) |
| **Video-Holmes Cross-Architecture** | **140 unique vids** | **`Qwen2.5-VL-7B-Instruct`** *(Pure Transformer)* | Multi-Step (`max_steps=6`) Acc | `22/140 (15.7%)` (`48.8f`, `2.05` calls) | — | `l_rt 24 (17.1%)` / **`l_full 40/140 (28.6%)`** (**`46.3f`**, **`1.89` calls −7.8%**, `rx=1.29s`) | **`lave_full` +24 / −6 (`+12.86 pp`, `p = 0.00143`)** with **−7.8% visual calls** & 0 vision replay | [score_qwen25vl_7b.json](results/video_holmes/score_qwen25vl_7b.json) |
| **Video-Holmes Thinking Mode** | 46 balanced prefix | **`Qwen3.5-9B-Thinking`** (`enable_thinking=True`) | Multi-Step (`max_steps=6`) Acc | `11/46 (23.9%)` (`95.0f`, `4.93` calls) | — | `l_rt 14 (30.4%)` / **`l_full 15/46 (32.6%)`** (**`89.0f`**, **`4.57` calls −7.3%**) | Strict monotonic ladder: `23.9% → 26.1% → 30.4% → 32.6%` (**`+8.70 pp`**, `l_full vs v_lat +3/−0`) | [score_qwen35_9b_thinking.json](results/video_holmes/score_qwen35_9b_thinking.json) |

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

### 5. Video-Holmes 140-Video Multi-Step VideoSeek + LAVE Evaluation (`results/video_holmes/`)

Frozen blind cohort (`140` questions across `140` unique videos, balanced `20` per each of all 7 categories, strictly disjoint from `pilot14`, single A40 GPU 1, released to `0 MiB`, pure single-video self-controls with zero cross-video foil, `385/385` controls passed, `0` vision calls in Receiver & Latent QA).

| Arm (`N = 140`, `max_steps = 6`) | Overall Acc | Rerouted (`N = 115`) | Paired vs `vanilla_text` | Avg Steps / Visual Calls / Frames |
| :--- | :---: | :---: | :---: | :---: |
| `vanilla_text` (Official VideoSeek) | `40 / 140` (`28.57%`) | `36 / 115` (`31.30%`) | — | `4.66` / `3.86` / `77.71` |
| `vanilla_latent_qa` (+ Mech B: Top-8 0-pixel verify) | `44 / 140` (`31.43%`) | `40 / 115` (`34.78%`) | `+12 / −8` (`+2.86 pp`, `p = 0.503`) | `77.71` *(0 extra vision calls)* |
| `lave_full` (+ Mech A Route + Mech B Verify) | `48 / 140` (`34.29%`) | `44 / 115` (`38.26%`) | `+21 / −13` (`+5.71 pp`, `p = 0.229`; vs `v_lat` `+14/−10`) | **`4.64` / `3.75` / `76.00`** |
| **`lave_route_text` (+ Mech A: LAVE Routing)** | **`50 / 140` (`35.71%`)** | **`46 / 115` (`40.00%`)** | **`+18 / −8` (`+7.14 pp`, 2-sided `p = 0.0755`, 1-sided `p = 0.0378`)** | **`4.64` / `3.75` / `76.00`** |

**Per-Category Breakdown (`n = 20` each)**:

| Arm | CTI | IMC | MHR | PAR | SR | TA | TCI | **Total (140)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla_text` | 4 | 7 | 8 | 6 | 7 | 4 | 4 | `40 (28.6%)` |
| `vanilla_latent_qa` | 5 | 8 | 8 | 6 | 5 | **6** | **6** | `44 (31.4%)` |
| **`lave_route_text`** | **5** | **9** | **11** | **8** | 6 | 5 | **6** | **`50 (35.7%)`** |
| `lave_full` | **5** | 8 | **11** | 5 | **9** | 5 | 5 | `48 (34.3%)` |

*(Anytime Step-4 checkpoint: `vanilla_step4 = 52/140 (37.1%)` degrades `−8.57 pp` to `28.6%` by Step 6 under late-step blind-search drift, whereas `lave_step4 = 49/140 (35.0%)` improves `+0.71 pp` to `35.7%` by Step 6. See [results/video_holmes/notes.md](results/video_holmes/notes.md).)*

**Direct-Visual (`VideoSeek-Image`), Equal-Budget (`Clock-Split`), & Unshared Standalone Latency Audit (`results/video_holmes/score_baselines_eval140.json`)**:

| Method (`N = 140` Video-Holmes, `Qwen3.5-9B`) | Accuracy (`N = 140`) | vs `vanilla_text` | Mean Frames (Fed to VLM) | Unshared Standalone Latency (`n = 28`) |
| :--- | :---: | :---: | :---: | :---: |
| `vanilla_text` (Caption VideoSeek) | `40 / 140` (`28.57%`) | baseline | `77.71` | `70.95 s` (`3.89` vcalls, `78.3f`) |
| **`videoseek_image`** *(Direct Raw-Pixel Interleaved)* | `40 / 140` (`28.57%`) | `+0.00 pp` | `75.31` (**`216.69` fed**, `44,953` prompt tok, `49.1` pruned) | `33.44 s` *(no captioner, severe visual context overflow)* |
| `clock_split_text` *(Equal-Budget Geometric 50/50)* | `48 / 140` (`34.29%`) | `+5.71 pp` | `77.94` | — |
| `clock_split_uniform_qa` *(Uniform 8-Frame 0-Pixel QA)* | `46 / 140` (`32.86%`) | `+4.29 pp` | `77.94` | — |
| **`lave_full`** *(Mech A Route + Mech B 0-Pixel Verify)* | **`48 / 140` (`34.29%`)** | **`+5.71 pp`** *(vs `VideoSeek-Image` `+5.71 pp`; vs `Clock-Uniform` `+1.43 pp`)* | **`76.00`** | `77.57 s` *(faster than `v_lat 78.33 s`)* |
| **`lave_route_text`** *(Mech A: LAVE Visual-State Routing)* | **`50 / 140` (`35.71%`)** | **`+7.14 pp`** *(vs `VideoSeek-Image` `+7.14 pp`; vs `Clock-Split` `+1.43 pp`)* | **`76.00`** | **`70.81 s` (FASTER than `vanilla_text`! `−18.3%` vcalls, `−14.6%` frames)** |

---

### 6. Video-MME 216 & Multi-Step LongVideoBench-150 (`results/videomme_216/`, `results/lvb150_multistep/`)

- **Video-MME 216** (`72` Short + `72` Medium + `72` Long, balanced across all 12 tasks & 6 domains, `score_mme.json` SHA `df6f2fad...`):
  - `vanilla_text`: `125/216 (57.87%)` (`Short 77.78% / Medium 55.56% / Long 40.28%`; `4.11` st / `3.33` vcalls / `69.26` f)
  - `lave_route_text` (+Mech A): `131/216 (60.65%, +2.78 pp)` (`79.17% / 61.11% / 41.67%`; **`3.88` st / `3.04` vcalls −8.7% / `64.67` f**)
  - **`lave_full` (+Mech A+B)**: **`138/216 (63.89%, +26/−13, +6.02 pp, p = 0.053)`** (`81.94% / 63.89% / 45.83%`; **`3.04` vcalls −8.7%**)
  - **`vanilla_latent_qa` (+Mech B)**: **`141/216 (65.28%, +23/−7, +7.41 pp, p = 0.00522)`** (`80.56% / 63.89% / 51.39%`)
- **LongVideoBench-150 Multi-Step** (`max_steps=6`, frozen `PIN 27b9d833...`, `score_lvb150_multistep.json` SHA `cf8d73d0...`):
  - `lave_route_text` **`87/150 (58.00%)`** vs `vanilla_text` `85/150 (56.67%)` (`+1.33 pp`), driven by **`+5.94 pp` on ultra-long 1-hour videos (`dg=3600`, `n=101`: `59.41% vs 53.47%`)**, cutting visual calls **`−6.0%`** (`4.73 vs 5.03`) and frames `−4.8` (`91.63 vs 96.43`).
- **Pooled Multi-Step Summary Across All 3 Benchmarks (`N = 506` Unique Long Videos, Frozen `Qwen3.5-9B`)**:
  - `vanilla_text`: `250 / 506 (49.41%)` → `lave_route_text`: **`268 / 506 (52.96%, +3.56 pp)`** → `vanilla_latent_qa`: **`269 / 506 (53.16%, +3.75 pp)`** → **`lave_full`: `270 / 506 (53.36%, +3.95 pp)`**, with **fewer steps, fewer visual calls, and fewer frames on 3/3 benchmarks**!

---

### 7. Native vLLM 0.19.0 Parity & Speedup Benchmark (`results/vllm_parity/bench_vllm_b33.json`)

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
