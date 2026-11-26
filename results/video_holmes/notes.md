# Video-Holmes Multi-Step VideoSeek + LAVE Evaluation (`pilot14` & `eval140`)

- **Completed**: 2026-11-26 20:29 CST (single A40 GPU 1, released cleanly to `0 MiB`, `gpu_released: true`)
- **Remote Dir**: `/data1/llj/lljmas/qwen35_holmes_videoseek_lave_20261126_01`
- **Runtime**: Isolated vLLM 0.19.0 environment (`/data1/llj/lljmas/envs/lave_vllm_20261126`), frozen `Qwen3.5-9B`, 100% Training-Free (`temperature=0.0`, `thinking=False`), `max_steps = 6`, zero vLLM source modifications, `forbid_vision_hook` enforcing `visual_forwards == 0` in Receiver and Latent Verification.
- **Receipts & SHA-256**:
  - `preflight_eval140.json`: `148f64d0e0745ff656bb28cda54654c950dee005bb5dc6df5ab67560501ccbbd` (`140` questions across `140` unique videos, `20` per each of all 7 categories, strictly disjoint from `pilot14`, `labels=0`)
  - `eval140.terminal.json`: `d9483268ba5841479a2b9b8050f02f90b1b0fcc9d8f4706940e4d6ac1dab947a`
  - `score_eval140.json`: `276bb198ff2108359fd5b218848f00dfda7f12e5855f838120322cf8f724c89f`

---

## 1. Main 140-Video Multi-Step Results (`max_steps = 6`, 7 Categories × 20)

| Arm (`N = 140`, `140` unique videos) | Overall Acc | Rerouted Subset (`N = 115`) | Paired vs `vanilla_text` (Exact McNemar) | Avg Steps / Visual Calls / Frames |
| :--- | :---: | :---: | :---: | :---: |
| `vanilla_text` (Official Multi-Step VideoSeek) | `40 / 140` (`28.57%`) | `36 / 115` (`31.30%`) | — | `4.66` / `3.86` / `77.71` |
| `vanilla_latent_qa` (+ Mech B: Top-8 Zero-Pixel Verify) | `44 / 140` (`31.43%`) | `40 / 115` (`34.78%`) | `+12 / −8` (`+2.86 pp`, `p = 0.503`) | `77.71` *(0 extra vision calls)* |
| `lave_full` (+ Mech A Route + Mech B Verify) | `48 / 140` (`34.29%`) | `44 / 115` (`38.26%`) | `+21 / −13` (`+5.71 pp`, `p = 0.229`; vs `v_lat` `+14/−10`, `+2.86 pp`) | **`4.64` / `3.75` / `76.00`** |
| **`lave_route_text` (+ Mech A: LAVE Visual-State Routing)** | **`50 / 140` (`35.71%`)** | **`46 / 115` (`40.00%`)** | **`+18 / −8` (`+7.14 pp`, 2-sided `p = 0.0755`, 1-sided `p = 0.0378`)** | **`4.64` / `3.75` / `76.00`** |

- **Single-Video Self-Controls** (zero cross-video foil): **`385 / 385` (`100%`)** scans passed all 3 single-video controls (`lift > 0`, `lift > scrambled`, `lift > median + 0.25`); `115 / 140` (`82.1%`) videos triggered `306` rerouted tool calls; Receiver latency averaged **`10.26 s`/video** (`0` vision encoder calls).
- **Anytime Step-4 Checkpoint (Honest Multi-Step Drift Analysis)**:
  - `vanilla_step4`: `52 / 140` (`37.14%`) → drops **`−8.57 pp`** down to `vanilla_text` `40 / 140` (`28.57%`) by Step 6 due to late-step blind-search distraction (`"No relevant content found."` / irrelevant clips).
  - `lave_step4`: `49 / 140` (`35.00%`) → **improves `+0.71 pp`** to `lave_route_text` `50 / 140` (`35.71%`) by Step 6, demonstrating robustness against multi-step evidence drift.

---

## 2. Per-Category Breakdown (`eval140`, `n = 20` per category)

| Category | `vanilla_text` | `vanilla_latent_qa` | **`lave_route_text`** | `lave_full` | `lave_route_text` vs `vanilla_text` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTI** (Clue & Theme Inference) | `4` (`20.0%`) | `5` (`25.0%`) | **`5` (`25.0%`)** | **`5` (`25.0%`)** | `+5.0 pp` (`+1/−0`) |
| **IMC** (Intention & Motive Chaining) | `7` (`35.0%`) | `8` (`40.0%`) | **`9` (`45.0%`)** | `8` (`40.0%`) | **`+10.0 pp`** (`+3/−1`) |
| **MHR** (Multi-Hop Reasoning) | `8` (`40.0%`) | `8` (`40.0%`) | **`11` (`55.0%`)** | **`11` (`55.0%`)** | **`+15.0 pp`** (`+4/−1`) |
| **PAR** (Plot & Action Reasoning) | `6` (`30.0%`) | `6` (`30.0%`) | **`8` (`40.0%`)** | `5` (`25.0%`) | **`+10.0 pp`** (`+5/−3`) |
| **SR** (Social Relationship) | `7` (`35.0%`) | `5` (`25.0%`) | `6` (`30.0%`) | **`9` (`45.0%`)** | `−5.0 pp` (`0/−1`; `lave_full` `+10.0 pp`) |
| **TA** (Temporal Analysis) | `4` (`20.0%`) | **`6` (`30.0%`)** | `5` (`25.0%`) | `5` (`25.0%`) | `+5.0 pp` (`+3/−2`) |
| **TCI** (Temporal Causal Inference) | `4` (`20.0%`) | **`6` (`30.0%`)** | **`6` (`30.0%`)** | `5` (`25.0%`) | **`+10.0 pp`** (`+2/−0`) |
| **Total (`N = 140`)** | `40` (`28.6%`) | `44` (`31.4%`) | **`50` (`35.7%`)** | `48` (`34.3%`) | **`+7.14 pp` (6/7 categories win!)** |
