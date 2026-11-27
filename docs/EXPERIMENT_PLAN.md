# LAVE2027 Master Experimental Plan (N >= 300 Per Cohort, Unified VideoSeek max_steps = 8)

> **Core Scientific Thesis**: In multi-step long-video tool-use agents, discarding precomputed visual representations after coarse captioning induces *both* (A) **routing myopia** (blind search in subsequent steps) and (B) **caption-compression loss** (dropping fine visual details before answering). **LAVE2027** resolves both bottlenecks **100% training-free** on frozen VLMs via zero-replay post-merger state capture, calibrated-lift zero-pixel routing (**Mech A**), and zero-pixel visual-state readout (**Mech B**) natively inside zero-patch `vLLM 0.19.0`.

---

## 0. Global Invariants & Zero-Waste Append-Only Design

1. **Unified Agent Budget**: Official multi-step `VideoSeek` (`overview 32f` / `skim 16f` / `focus 16f` / `answer`), **`max_steps = 8`** across **all** main and ablation tables, with zero-cost anytime `Step-4` and `Step-6` checkpoints recorded inside every 8-step trajectory.
2. **Minimum Statistical Power (`N >= 300`)**: Every primary benchmark cohort and every ablation table contains **`N >= 300`** matched samples (`315 / 306 / 306 / 300`), strictly blind (`labels = 0`), single-unseal CPU scored, with exact two-sided McNemar `p`-values.
3. **Strict Matched-Cohort Ablations**: Every row inside a given ablation table is evaluated on the **exact same `N = 315` cohort (`preflight_holmes315.json`)** — zero sample-size mismatches across rows.
4. **Zero Duplicate Compute (Append-Only Manifests)**:
   - `preflight_holmes315.json` (`N = 315`, 100% unique videos, `45 × 7` cats): indices `[0..139]` are bit-for-bit identical to frozen `preflight_eval140.json` + `175` brand-new unseen videos (disjoint from `pilot14`).
   - `preflight_mme306.json` (`N = 306`, 100% unique videos, `102S / 102M / 102L`, 12 tasks): indices `[0..215]` are bit-for-bit identical to frozen `preflight_mme.json` + `90` brand-new unseen videos.
   - `preflight_lvb306.json` (`N = 306` long videos `dg >= 600`, 17 cats): indices `[0..149]` are bit-for-bit identical to frozen `preflight_lvb150.json` + `156` brand-new disjoint long-video questions (prioritizing all 64 brand-new unseen videos).
   - `preflight_cgbench300.json` (`N = 300` unseen long videos with GT clue intervals): indices `[0..95]` are bit-for-bit identical to frozen `Confirm-96` + `204` brand-new unseen CG-Bench videos.
5. **100% Fair Evaluation**: Deterministic `raw_decode` + positional/NL tool parser (`~0` invalid tool calls) and zero-refusal `allowed_token_ids` forced-choice fallback (`invalid_answers = 0` across all arms), `temperature = 0.0`, `visual_forwards == 0` strictly enforced by `forbid_vision_hook` inside Receiver and Latent Readout.

---

## 1. Main Experiment Table (Table 1 — Breadth Across Benchmarks & Model Families, All `N >= 300`, `max_steps = 8`)

**Logical Claim (H1)**: *Training-free visual-state communication simultaneously improves answer accuracy AND reduces visual tool calls & sampled frames across diverse long-video tasks and distinct model architectures.*

### 1a. Directly Reusable Published Baselines (Cited Directly — 0 GPU Cost!)

| Method | Category | Backbone | Video-MME (w/o sub) | LongVideoBench | Video-Holmes | Verified Source & Citation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `Qwen2.5-VL-7B` (Uniform) | Single-Turn VLM | Qwen2.5-VL-7B | `65.1%` | `56.0%` | `34.7%` | Bai et al. (2025); cited in Video-o3 (ICML) |
| `LLaVA-Video-7B` | Single-Turn VLM | Qwen2-7B | `63.3%` | `58.2%` | — | Zhang et al. (2024); cited in Video-o3 |
| `VideoChat-R1-7B` | RL Post-Trained | Qwen2.5-VL-7B | — | — | `33.0%` (`32f`) | Li et al. (2025); VideoSeek Table 2 (`arXiv:2603.20185`) |
| `SEED-Bench-R1-7B` | RL Post-Trained | Qwen2.5-VL-7B | — | — | `33.5%` (`32f`) | Chen et al. (2025); VideoSeek Table 2 (`arXiv:2603.20185`) |
| `Video-R1-7B` | RL Post-Trained | Qwen2.5-VL-7B | `61.4%` | — | `36.5%` (`32f`) | Feng et al. (2025); VideoSeek Table 2 & Video-o3 |
| `VideoMind-7B` | Trained LoRA Agent | Qwen2.5-VL-7B | `58.2%` (`L: 49.2%`) | `56.3%` | — | Liu et al. (2025) official GitHub / paper |
| `Video-RTS-7B` / `Video-MTR-7B` | Iterative Agent | Qwen2.5-VL-7B | `63.0%` / `59.0%` | `56.6%` / `56.4%` | — | Cited in Video-o3 (ICML) |
| `Conan-7B` | Multi-Turn Tool Agent (Trained) | Qwen2.5-VL-7B | `60.5%` | `56.6%` | `44.6%` | Cited in Video-o3 (ICML) |
| `Video-o3-7B` (SFT+RL, 573k, ≤768f) | Trained Interleaved Agent | Qwen2.5-VL-7B | `66.5%` | `60.5%` | `46.5%` | Video-o3 (ICML) |
| `VideoAgent` / `VideoTree` / `VCA` | Training-Free Agents | Proprietary/LLM | `L: 46.4% / 53.1% / 54.2%` | — | — | VideoSeek Table 1 (`arXiv:2603.20185`) |
| `VideoSeek` (GPT-5, Reference Upper) | Training-Free Agent | GPT-5 | `L: 70.1%` (`60.9f`) | `73.5%` (`29.6f`) | `47.3%` (`42.7f`) | Lin et al. (CVPR 2026, `arXiv:2603.20185`) |

> **Important Note on `Qwen2.5-VL-7B + VideoSeek`**: Official VideoSeek (`arXiv:2603.20185`) only evaluated proprietary reasoning backbones (`GPT-5`, `o4-mini`, `GPT-4.1`), and LARVA (`ICLR 2027` withdrawn) used a 20k-trajectory CoT-distilled LoRA student rather than off-the-shelf `Qwen2.5-VL-7B + VideoSeek`. Therefore, **same-cohort paired `Vanilla VideoSeek` MUST be executed alongside `VideoSeek + LAVE`** (sharing Step-1 `overview` + `tool_cache` so marginal overhead is only ~35–40%, while unlocking exact paired McNemar `p`-values and `LAVE-Readout`!).

### 1b. Controlled Same-Harness Internal Comparison (Executed on Exact Frozen `N >= 300` Cohorts)

| Backbone Family | Architecture | Benchmarks (All `N >= 300`, `max_steps = 8`) | Arms Evaluated | Primary Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **`Qwen2.5-VL-7B-Instruct`** *(LARVA's exact backbone)* | Pure Transformer | • **Video-Holmes-315** (`N=315`)<br>• **Video-MME-306** (`N=306`: 102S/102M/102L)<br>• **LongVideoBench-306** (`N=306`, 17 cats) | `Vanilla VideoSeek (Text)`<br>`LAVE-Route (Mech A)`<br>`LAVE-Readout (Mech B)`<br>`LAVE-Full (Mech A+B)` | Accuracy, Paired McNemar `p`, Avg Steps, Visual Calls, Frames, `rx_s` |
| **`Qwen3.5-9B`** *(Modern Hybrid VLM)* | Hybrid Gated DeltaNet + Attention | • **Video-Holmes-315** (`N=315`)<br>• **Video-MME-306** (`N=306`: 102S/102M/102L)<br>• **LongVideoBench-306** (`N=306`, 17 cats) | Same 4 arms + `VideoSeek-Image` (on Holmes-315) | Same exact metrics (zero-duplicate resume on `[0..N_old]` where `max_steps` matches) |

---

## 2. Core Ablation Table (Table 2 — Strict Causal Attribution on Exact Same `N = 315` Cohort, `Video-Holmes-315`, `max_steps = 8`)

**Logical Claim (H2 — Preempting LARVA Reviewer `BgQn W1` & `vkFe W2`)**: *Every component of LAVE contributes causally, and gains cannot be explained away by frame diversification (`Clock-Split`), direct raw pixels (`VideoSeek-Image`), uncalibrated prior bias, or random/uniform bank sampling.*

All 10 rows share the **EXACT SAME 315 videos (`preflight_holmes315.json`)** and **`max_steps = 8`**:

| # | Variant | Routing Source | Self-Control Gate | Calibration | Multi-Peak | Readout Interface | Causal Question Isolated |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | `Vanilla VideoSeek (Text)` | Planner text | — | — | — | Pure text | Caption-only multi-step baseline |
| 2 | **`VideoSeek-Image` (Direct Pixels)** | Planner raw pixels | — | — | — | Direct raw frames | Direct-visual alternative (Reviewer `vkFe W2`) |
| 3 | **`Clock-Split` (Equal-Budget 50/50)** | Largest unvisited midpoint | — | — | Fixed 1 | Pure text | Equal-budget diversification without visual states |
| 4 | **`Clock-Split + Uniform-8 Readout`** | Geometric 50/50 | — | — | Fixed 1 | Uniform-8 cached states | Uniform state sampling vs LAVE lift-ranked Mech B |
| 5 | `LAVE w/o Zero-Calibration` | Raw `z_Y - z_N` | 3-Control | **Off** | Entropy | Pure text / Top-8 | Necessity of zero-embedding calibration across resolutions |
| 6 | `LAVE w/o Self-Control Gate` | Calibrated lift | **Off** (always split) | On | Entropy | Pure text / Top-8 | Necessity of single-video 3-control gate |
| 7 | `LAVE Fixed K=1 (No Entropy)` | Calibrated lift | 3-Control | On | **Fixed K=1** | Pure text / Top-8 | Necessity of entropy-guided multi-interval allocation |
| 8 | **`LAVE-Route (Mech A Only)`** | Calibrated lift | 3-Control | On | Entropy | Pure text | Clean isolation of Mechanism A (routing only) |
| 9 | **`LAVE-Readout (Mech B Only)`** | Planner text | — | On | — | Top-8 lift states | Clean isolation of Mechanism B (zero-pixel readout only) |
| 10 | **`LAVE-Full (Mech A + Mech B)`** | Calibrated lift | 3-Control | On | Entropy | Top-8 lift states | Full training-free synergy |

*Zero-Extra-Vision Sensitivity Sweep (same `N = 315` cached banks)*: Readout budget `M ∈ {4, 8, 16}` frames (`0` vision calls, tests accuracy–context trade-off of Mech B).

---

## 3. Auxiliary Experiments (Tables 3–6 — Closing Every Remaining Reviewer Angle)

### Aux-1 (Table 3): Ground-Truth Evidence-Acquisition Quality Independent of QA (`CG-Bench-300`, `N = 300` Unseen Long Videos)
- **Why**: Directly addresses LARVA Reviewer `BgQn S2` ("evidence quality beyond raw accuracy"), replacing LARVA's 100-question manual human judge study with **300 objective GT clue-interval evaluations** (`Confirm-96` prefix already at `42.7% vs 27.1%, p = 0.000729`, expanded append-only to `N = 300`).
- **Metrics**: New GT Clue-Hit Rate (excl. Source), Empty-Report Rate (`"No relevant content found."`), `Text` vs `Clock-Split` vs `LAVE (State-Split)`.

### Aux-2 (Table 4): Cross-Scale & Thinking-Model Generalization (Matched `Video-Holmes-315`, `N = 315`, `max_steps = 8`)
- **Why**: Directly addresses LARVA Reviewers `vkFe W3` & `BgQn Q5` ("does it generalize to smaller/larger scales and dedicated Thinking models?").
- **Rows (all matched on `Video-Holmes-315`)**:
  - `Qwen3.5-4B` (Small scale — already showed `+12.14 pp, p = 0.0015` on `140`, beating 9B text!)
  - `Qwen2.5-VL-7B-Instruct` (Pure Transformer — LARVA's backbone)
  - `Qwen3.5-9B` (Standard hybrid 9B)
  - `Qwen3.5-9B-Thinking` (`enable_thinking=True` — already showed strict monotonic `+8.70 pp` on 46-video prefix!)

### Aux-3 (Table 5): Horizon Scaling & Multi-Step Drift Curve (`Step-4 → Step-6 → Step-8`, Zero Extra GPU Cost!)
- **Why**: Demonstrates why longer reasoning horizons (`max_steps = 8`) hurt blind text search (late-step drift) while LAVE visual-state grounding stabilizes or improves monotonically across Steps `4 → 6 → 8`.
- **Sample Size**: Automatically collected across **all `N = 315 / 306 / 306` trajectories** with zero extra visual tool calls.

### Aux-4 (Table 6): Exact Unshared End-to-End Latency, Memory, & Token Accounting (`vLLM 0.19.0`, Zero Mismatches)
- **Why**: Directly addresses LARVA Reviewer `BgQn W3 / Q4` (strictly consistent checkpoint & end-to-end timing boundaries).
- **Sub-table 6a**: HF vs Native `vLLM 0.19.0` parity & `3.35×` Receiver speedup (`bench_vllm_b33.json`, `0` source patches, `0` vision calls verified).
- **Sub-table 6b**: Unshared standalone end-to-end timing (`n >= 60` balanced subset of `Video-Holmes-315`, zero shared overview, zero `tool_cache`, single A40 GPU, batch=1, BF16):
  - Reports stage-by-stage wall-clock latency (`Overview`, `Skim/Focus`, `Planner`, `Receiver 0-Pixel`, `Readout 0-Pixel`, **Total s**), Vision Calls, Acquired Frames, **Total Frames Fed to VLM** (`VideoSeek-Image` `~217f / 45k prompt tokens` vs LAVE `~70f`), and Peak VRAM.

### Aux-5 (Figure 4 & Appendix): Self-Control Signal Audit + Diverse Qualitative Success & Failure Gallery
- **Why**: Directly addresses Reviewer `fjeb W2` & internal validity of single-video self-controls.
- **Content**:
  1. Distribution of `lift(real)` vs `lift(zero)` vs `lift(scrambled)` vs `median + δ` across `1,000+` real routing decisions.
  2. 6-case qualitative gallery: (i) Multi-hop clue discovery, (ii) Fine-grained on-screen text rescue, (iii) 1-hour temporal jump, (iv) Step 4→8 drift prevention, (v) `VideoSeek-Image` context overflow vs LAVE bank, and (vi) Honest failure case (dense counting).

---

## 4. Execution Schedule (Strictly Single GPU Right Now — GPU 0 Only!)

1. **Stage 1 (ACTIVE NOW on GPU 0 ONLY, Zero Duplicate Resume)**:
   - `Qwen2.5-VL-7B-Instruct` (`max_steps = 8`, `0` invalid tools, `0` refusals):
     - `holmes315` (`N = 315`, resuming seamlessly from `case_037+`!)
     - → `mme306` (`N = 306`: `102S / 102M / 102L`)
     - → `lvb306` (`N = 306`: all 17 categories, `dg >= 600`)
2. **Stage 2 (Next on GPU 0, or Multi-GPU Overnight if Approved)**:
   - `Qwen3.5-9B` (`max_steps = 8`) on the exact same `holmes315`, `mme306`, `lvb306` manifests.
3. **Stage 3 (Table 2 Matched `N = 315` Ablations on `Video-Holmes-315`, `max_steps = 8`)**:
   - `VideoSeek-Image`, `Clock-Split`, `Clock-Split + Uniform-8`, `No-Calibration`, `No-Gate`, `Fixed K=1`, `M ∈ {4, 8, 16}` sweep, + `n = 60` unshared latency audit.
4. **Stage 4 (Aux-1 & Aux-2)**:
   - `CG-Bench-300` GT clue-hit expansion (`[0..95]` exact `Confirm-96` reused + `204` new unseen videos).
   - `Qwen3.5-4B` & `Qwen3.5-9B-Thinking` on `Video-Holmes-315`.
