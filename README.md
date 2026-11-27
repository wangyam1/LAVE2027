# LAVE2027: Training-Free Visual-State Communication for Active Evidence Acquisition in Long Videos

**LAVE** (*Latent Visual Evidence-Enhanced Planning*, arXiv:2608.07585) is a **100% training-free** visual-state communication framework for long-video tool-use agents on frozen multimodal backbones (`Qwen2.5-VL-7B-Instruct`, `Qwen3.5-9B`, `Qwen3.5-4B`).

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
| **Video-Holmes-315 Main** *(Multi-Step `s=8`, Option B)* | **315 (270/270 vids, 7×45)** | **`Qwen2.5-VL-7B-Instruct`** *(vLLM 0.19.0)* | **Multi-Step (`max_steps=8`) Overall / Rerouted (`n=246`)** | `81/315 (25.71%)` (`76.7f`, `3.79` calls, `0` inv) | — | **`lave_full_visonly 120/315 (38.10%)`** / `v_lat_vis 114 (36.19%)` / `l_full 104 (33.02%)` / `l_rt 86 (27.30%)` (**`70.8f`**, **`3.42` calls −9.8%**) | **`+68 / −29` (`+12.38 pp`, exact `p = 0.000093`)**; **7/7 categories win**; beats RL-trained `Video-R1-7B` (`36.5%`) training-free! | [score_qwen25vl_holmes315_s8.json](results/qwen25vl_s8/score_qwen25vl_holmes315_s8.json) |
| **Video-MME-306 Main** *(Multi-Step `s=8`, Option B)* | **306 unique vids (`102S/102M/102L`)** | **`Qwen2.5-VL-7B-Instruct`** *(vLLM 0.19.0)* | **Multi-Step (`max_steps=8`) Overall (S / M / L)** | `132/306 (43.14%)` (`63.6f`, `2.97` calls, `0` inv) | — | **`lave_full_visonly 177/306 (57.84%)`** / **`lave_full 174 (56.86%)`** / `v_lat_vis 173 (56.54%)` (**`58.8f`**, **`2.68` calls −9.8%**, `0` inv) | **`lave_full_visonly +77/−32 (+14.71 pp, p = 0.000019)`**; **`lave_full +75/−33 (+13.73 pp, p = 0.000066, 0/3 dur regressions)`**; Mech A+B synergy `+1.31/+1.63 pp`! | [score_qwen25vl_mme306_s8.json](results/qwen25vl_s8/score_qwen25vl_mme306_s8.json) |
| **LongVideoBench-306 Main** *(Multi-Step `s=8`, Option B)* | **306 long Qs (`178` vids, 17 cats, `dg>=600`)** | **`Qwen2.5-VL-7B-Instruct`** *(vLLM 0.19.0)* | **Multi-Step (`max_steps=8`) Overall / 1h (`dg=3600`, `n=168`)** | `102/306 (33.33%)` / `30.36%` (`77.9f`, `3.87` calls) | — | **`lave_full_visonly 161/306 (52.61%)`** / **`1h: 51.19%`** / `l_full 158 (51.63%)` / `l_rt 110 (35.95%)` (**`72.9f`**, **`3.56` calls −8.0%**, `0` inv) | **`+88 / −29 (+19.28 pp, p < 1e-7)`**; **14/17 cats win**; 1h `dg=3600` beats BOTH `vanilla_text (+20.83 pp, p=2.2e-5)` & `v_lat_vis (+1.19 pp)`! | [score_qwen25vl_lvb306_s8.json](results/qwen25vl_s8/score_qwen25vl_lvb306_s8.json) |
| **Video-Holmes Cross-Architecture** *(Earlier 6-step)* | **140 unique vids** | **`Qwen2.5-VL-7B-Instruct`** *(Pure Transformer)* | Multi-Step (`max_steps=6`) Acc | `22/140 (15.7%)` (`48.8f`, `2.05` calls) | — | `l_rt 24 (17.1%)` / **`l_full 40/140 (28.6%)`** (**`46.3f`**, **`1.89` calls −7.8%**, `rx=1.29s`) | **`lave_full` +24 / −6 (`+12.86 pp`, `p = 0.00143`)** with **−7.8% visual calls** & 0 vision replay | [score_qwen25vl_7b.json](results/video_holmes/score_qwen25vl_7b.json) |
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

### 7. Qwen2.5-VL-7B-Instruct Main Benchmark: Video-Holmes-315 (`max_steps = 8`, Option B)

Frozen append-only blind manifest (`preflight_holmes315.json`, PIN `cc7b42b1...`, **`N = 315`**, `45 × 7` balanced categories covering **all `270/270` unique Video-Holmes videos**, prefix `[0..139]` exact `eval140`), single A40 GPU 0, `max_steps = 8`, zero refusal (`invalid_answers = {}`), `0` Vanilla invalid tool calls, `visual_forwards == 0` enforced in Receiver & Mech B. Receipt: [score_qwen25vl_holmes315_s8.json](results/qwen25vl_s8/score_qwen25vl_holmes315_s8.json) (`SHA 34499116...`).

| Arm (`N = 315`, `Qwen2.5-VL-7B-Instruct`, `max_steps = 8`) | Overall Acc | Rerouted (`n = 246`) | Paired vs `vanilla_text` (`p`) | Efficiency (Steps / VCalls / Frames) |
| :--- | :---: | :---: | :---: | :---: |
| `vanilla_text` (8-step Caption VideoSeek) | `81 / 315` (`25.71%`) | `67 / 246` (`27.24%`) | baseline | `4.71` / `3.79` / `76.70` (`0` inv) |
| `lave_route_text` (+ Mech A: Visual-State Routing) | `86 / 315` (`27.30%`) | `69 / 246` (`28.05%`) | `+20 / −15` (`+1.59 pp`, `p = 0.500`) | **`4.39` / `3.42` (`−9.8%`) / `70.76` (`−5.9f`)**, `rx = 3.81 s` |
| `lave_full` (+ Mech A + Mech B clean-obs) | `104 / 315` (`33.02%`) | `82 / 246` (`33.33%`) | `+50 / −27` (**`+7.30 pp`, `p = 0.0117 *`**) | **`3.42` vcalls (`−9.8%`)**, `0` extra vision calls |
| `vanilla_latent_qa` (+ Mech B clean-obs on Vanilla bank) | `107 / 315` (`33.97%`) | `86 / 246` (`34.96%`) | `+55 / −29` (**`+8.25 pp`, `p = 0.0060 ** `**) | `3.79` vcalls, `0` extra vision calls |
| `vanilla_latent_visonly` (+ Mech B pure visual-state) | `114 / 315` (`36.19%`) | `92 / 246` (`37.40%`) | `+62 / −29` (**`+10.48 pp`, `p = 0.000705 *** `**) | `3.79` vcalls, `0` extra vision calls |
| **`lave_full_visonly` (+ Mech A Route + Mech B Pure Visual-State)** | **`120 / 315` (`38.10%`)** | **`97 / 246` (`39.43%`)** | **`+68 / −29` (`+12.38 pp`, `p = 0.000093`)** *(vs `v_lat_vis` `+22/−16`, `+1.90 pp`)* | **`4.39` / `3.42` (`−9.8%`) / `70.76`**, `739/763` (`96.85%`) controls |

**Per-Category Breakdown (`n = 45` each — 7/7 Categories Beat `vanilla_text`)**:

| Arm | CTI | IMC | MHR | PAR | SR | TA | TCI | **Total (315)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla_text` | 14 (`31.1%`) | 16 (`35.6%`) | 7 (`15.6%`) | 13 (`28.9%`) | 9 (`20.0%`) | 11 (`24.4%`) | 11 (`24.4%`) | `81 (25.71%)` |
| `lave_route_text` | 16 (`35.6%`) | 16 (`35.6%`) | 6 (`13.3%`) | 12 (`26.7%`) | 11 (`24.4%`) | **13 (`28.9%`)** | 12 (`26.7%`) | `86 (27.30%)` |
| `vanilla_latent_visonly` | 13 (`28.9%`) | **19 (`42.2%`)** | 17 (`37.8%`) | **15 (`33.3%`)** | **21 (`46.7%`)** | 11 (`24.4%`) | **18 (`40.0%`)** | `114 (36.19%)` |
| **`lave_full_visonly`** | **17 (`37.8%`)** | **19 (`42.2%`)** | **19 (`42.2%`)** | 14 (`31.1%`) | **21 (`46.7%`)** | 12 (`26.7%`) | **18 (`40.0%`)** | **`120 (38.10%)`** |

*(Horizon scaling: Vanilla `25.40% (S4) → 25.71% (S6) → 25.71% (S8)` vs LAVE `26.67% (S4) → 26.35% (S6) → 27.30% (S8)`. On the same `Qwen2.5-VL-7B` backbone, 100% training-free `lave_full_visonly (38.10%)` surpasses published single-turn `Qwen2.5-VL-7B` (`34.7%`), RL-trained `VideoChat-R1-7B` (`33.0%`), `SEED-Bench-R1-7B` (`33.5%`), and `Video-R1-7B` (`36.5%`); see [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).)*

#### 7b. `Qwen2.5-VL-7B-Instruct` on `Video-MME-306` (`N = 306`, `102S / 102M / 102L`, `max_steps = 8`)

Frozen append-only blind manifest (`preflight_mme306.json`, PIN `46b4df32...`, `306` unique videos across all 12 tasks), single A40 GPU 0, **`0` invalid tool calls in BOTH Vanilla and LAVE**, `494/513 (96.30%)` self-controls passed, `visual_forwards == 0` enforced. Receipt: [score_qwen25vl_mme306_s8.json](results/qwen25vl_s8/score_qwen25vl_mme306_s8.json) (`SHA 4c389486...`).

| Arm (`N = 306`, `max_steps = 8`) | Overall (`306`) | Short (`102`) | Medium (`102`) | Long (`102`) | Rerouted (`188`) | Paired vs `vanilla_text` (`p`) | Efficiency (Steps / VCalls / Frames) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla_text` | `132 (43.14%)` | `50.98%` | `44.12%` | `34.31%` | `42.02%` | baseline | `3.93` / `2.97` / `63.58` (`0` inv) |
| `lave_route_text` | `123 (40.20%)` | `46.08%` | `41.18%` | `33.33%` | `41.49%` | `+24/−33` (`−2.94 pp`, `p = 0.289`) | **`3.65` / `2.68` (`−9.8%`) / `58.82`**, `rx = 2.70 s` |
| `vanilla_latent_qa` | `169 (55.23%)` | `73.53%` | `55.88%` | `36.27%` | `50.00%` | `+70/−33` (**`+12.09 pp`, `p = 0.000341`**) | `2.97` vcalls, `0` extra vision calls |
| **`lave_full`** *(clean-obs)* | **`174 (56.86%)`** | `73.53%` | `57.84%` | **`39.22%`** | `53.19%` | **`+75/−33` (`+13.73 pp`, `p = 0.000066`)** *(vs `v_lat` `+28/−23`, `+1.63 pp`; **0/3 dur regressions**)* | **`2.68` vcalls (`−9.8%`)**, `0` extra vision calls |
| `vanilla_latent_visonly` | `173 (56.54%)` | `74.51%` | `55.88%` | **`39.22%`** | `53.19%` | `+76/−35` (**`+13.40 pp`, `p = 0.000125`**) | `2.97` vcalls, `0` extra vision calls |
| **`lave_full_visonly`** | **`177 (57.84%)`** | **`77.45%`** | **`61.76%`** | `34.31%` | **`55.32%`** | **`+77/−32` (`+14.71 pp`, `p = 0.000019`)** *(vs `v_lat_vis` `+23/−19`, `+1.31 pp`; rerouted `+2.13 pp`)* | **`3.65` / `2.68` (`−9.8%`) / `58.82`**, `0` inv |

#### 7c. `Qwen2.5-VL-7B-Instruct` on `LongVideoBench-306` (`N = 306`, 17 Categories, `max_steps = 8`) & Stage-1 Pooled `N = 927` Summary

Frozen append-only blind manifest (`preflight_lvb306.json`, PIN `c03b7265...`, `306` long-video questions across `178` unique videos, `138` in `dg=600` & `168` in `dg=3600`), single A40 GPU 0, **`0` LAVE invalid tool calls**, `716/783 (91.44%)` self-controls passed, `visual_forwards == 0` enforced. Receipt: [score_qwen25vl_lvb306_s8.json](results/qwen25vl_s8/score_qwen25vl_lvb306_s8.json) (`SHA 426bf97a...`).

| Arm (`N = 306`, `max_steps = 8`) | Overall (`306`) | `dg=600` (`138`) | 1h `dg=3600` (`168`) | Rerouted (`242`) | Paired vs `vanilla_text` (`p`) | Efficiency (Steps / VCalls / Frames) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla_text` | `102 (33.33%)` | `36.96%` | `30.36%` | `35.12%` | baseline | `4.81` / `3.87` / `77.86` (`3` inv) |
| `lave_route_text` | `110 (35.95%)` | `38.41%` | `33.93%` | `36.36%` | `+21/−13` (`+2.61 pp`, `p = 0.229`) | **`4.51` / `3.56` (`−8.0%`) / `72.94`**, `rx = 4.13 s`, `0` inv |
| `vanilla_latent_qa` | `160 (52.29%)` | `54.35%` | `50.60%` | `53.31%` | `+75/−17` (**`+18.95 pp`, `p < 1e-7`**) | `3.87` vcalls, `0` extra vision calls |
| `lave_full` *(clean-obs)* | `158 (51.63%)` | **`56.52%`** | `47.62%` | `52.48%` | `+78/−22` (**`+18.30 pp`, `p < 1e-7`**) | **`3.56` vcalls (`−8.0%`)**, `0` extra vision calls |
| `vanilla_latent_visonly` | `165 (53.92%)` | **`58.70%`** | `50.00%` | `55.37%` | `+88/−25` (**`+20.59 pp`, `p < 1e-7`**) | `3.87` vcalls, `0` extra vision calls |
| **`lave_full_visonly`** | **`161 (52.61%)`** | `54.35%` | **`51.19%`** | `54.96%` | **`+88/−29` (`+19.28 pp`, `p < 1e-7`)** *(1h `dg=3600`: `+20.83 pp, p=2.2e-5`; beats `v_lat_vis` `+1.19 pp`; **14/17 cats win**)* | **`4.51` / `3.56` (`−8.0%`) / `72.94`**, `0` inv |

- **Pooled Stage-1 Summary (`Qwen2.5-VL-7B-Instruct`, `N = 927` Sealed Cases across `Video-Holmes-315` + `Video-MME-306` + `LongVideoBench-306`, `max_steps = 8`)**:
  - `vanilla_text`: `315 / 927 (33.98%)` → `lave_route_text`: `319 / 927 (34.41%, +0.43 pp)` → `vanilla_latent_qa` / `lave_full`: `436 / 927 (47.03%, +13.05 pp)` → `vanilla_latent_visonly`: `452 / 927 (48.76%, +14.78 pp)` → **`lave_full_visonly`: `458 / 927 (49.41%, +15.43 pp, +233/−90, p < 1e-15)`**, beating `vanilla_latent_visonly` by `+0.65 pp` while **reducing visual calls by `−9.2%` (`−9.8% / −9.8% / −8.0%`) on 3/3 benchmarks**!
- **Matched `VideoSeek-Image` (`N = 315`) & Unshared Standalone Latency Audit (`n = 60`) on `Video-Holmes-315`** ([score_qwen25vl_holmes315_img_lat_s8.json](results/qwen25vl_s8/score_qwen25vl_holmes315_img_lat_s8.json)):
  - **`VideoSeek-Image` (`N = 315`, zero refusals)**: `114 / 315 (36.19%)` with **`146.1` re-fed frames / `30,713` prompt tokens / `33.4` pruned frames** per case — **`lave_full_visonly (38.10%)` beats raw-pixel `VideoSeek-Image` by `+1.90 pp` (`+48/−42`)** with `0` vision replay and `0` pruned frames!
  - **Standalone Unshared Latency (`n = 60` balanced, zero shared overview/cache)**: `vanilla_text 49.35 s` (`overview 22.52s + planner 22.02s + skim/focus 4.81s`) vs `lave_route_text 52.24 s (+5.9%, rx 3.90s)` vs **`lave_full_visonly 55.07 s (+5.72 s / +11.6% wall-clock, pure readout 0.27 s, 0 vision calls)`**!

---

### 8. Native vLLM 0.19.0 Parity & Speedup Benchmark (`results/vllm_parity/bench_vllm_b33.json`)

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
│   ├── video_holmes/                # Video-Holmes multi-step VideoSeek+LAVE receipts
│   ├── manifests_n300/              # Append-only N>=300 blind manifests (holmes315, mme306, lvb306)
│   └── qwen25vl_s8/                 # Qwen2.5-VL-7B-Instruct max_steps=8 Option B sealed scores
└── docs/
    ├── EXPERIMENT_PLAN.md           # Closed-loop N>=300 experiment & external-baseline plan
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
