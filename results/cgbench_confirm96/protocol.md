# Frozen Independent Confirmation Protocol — 96 Sealed CG-Bench Long Videos

- **Cohort**: 96 independent long CG-Bench videos (`min_duration_s = 621.0`, `max_duration_s = 4163.0`, 1 primary `q1` question per unique video, strictly disjoint from all 228 previously exposed / reserved videos).
- **Pinned Vault**: `/data1/llj/lljmas/qwen35_cgbench_long_confirm_assets_20261121_01`
  - `roster.private.json`: SHA-256 `3c7d22e0e461529a8aa699f4b3c0921677fcf008c5c80bb7f6d1f0214749553a`
  - `qid_roles.private.json`: SHA-256 `8179e40ca3dc3fab7aaab63bbfa73b03063380ec638f412770d998d019142ce0`
  - `receipts/first20261121/integrity.private.json`: SHA-256 `87ed33eb745f2475201f33c52162c0f68d2eb0f30260ef1d1f708d1906ef4938`
  - `cgbench.json`: SHA-256 `f2db9d2e8144e30806c3eab3ea50e7e1681bf9dd31d0dd6ce2871f960559de12`
  - `metadata_projection_scanner.py`: SHA-256 `7b772076a62e55e3da090dc3bc4d6e62223b3a8716bb78f75b2d98684ac9cf20`

## Frozen Method (Zero Changes After Dev Lock)
1. **Training-Free & Zero Pixel Replay**: Frozen `Qwen3.5-9B`, `do_sample=False`, `thinking=False`, `seed=42`, 0 parameter updates, 0 pixel inputs to Planner or Receiver.
2. **Live Source Capture**: Decode 32 uniform overview frames (`np.linspace(0, n-1, 32)`). Run native `source_shared` ONCE while capturing 32 post-merger BF16 visual states (`capture_source`, zero Source replay) and native `source_query` ONCE (`source_cap=2048`).
3. **Common Text Planner**: Pure-text native `planning_messages` (`T_plus`) with `focus` schema (`planner_cap=512`). Invalid actions remain common skips across all three arms.
4. **Entropy-Guided Multi-Gap Receiver (`batch_size=1`, `use_cache=False`)**:
   - Deterministic `target_from_question(question)` and label-blind cross-video `wrong_target` foil.
   - Scan 32 captured Source states + zero baseline (33 forwards, 0 vision encoder calls).
   - Nominate top eligible Source gap with `lift > 0` (no uncalibrated raw `margin > 0` cutoff).
   - Require strict win over ALL THREE content controls: same-video half-turn `shifted`, channel-permuted/sign-flipped `scrambled`, and `wrong_target` (+3 forwards).
   - When approved, compute normalized Shannon entropy `H_norm` of softmax gap lifts across eligible gaps, choose `K in {1, 2, 4}` via textbook tertiles (`< 1/3 -> 1`, `< 2/3 -> 2`, `>= 2/3 -> 4`), select `K` non-adjacent NMS gaps (`|a-b| > 1`), and allocate `16` anchor frames + `K * (16 // K) = 16` latent frames (`32` total).
   - Matched `clock_split`: greedy farthest-point max-novelty selection with exact same `K`, `16 // K` frames/gap, NMS rule, and 16-frame text anchor.
5. **Focus-1 (100% Unmodified Native Focus Prompt)**:
   - Execute `text`, `state_split`, `clock_split` in counterbalanced order (`SCHEDULES[ordinal % 6]`) with exact-input deduplication and verified 32-frame / visual-token parity.
   - Run unmodified pure-text QA (`qa_cap=16`) on Focus-1 native observation.
6. **Focus-2 (100% Unmodified Native Focus Prompt & Original `text_action["query"]`)**:
   - Extract non-empty timestamp lines (`\d+(?:\.\d+)?s:`) from each arm's Focus-1 native observation.
   - Densest `<= 30s` timestamp cluster padded by `+-10s` (min `20s`) -> 16 unseen dense frames (`on_policy`).
   - Whenever any arm triggers Focus-2 on a video, arms without Focus-1 timestamps also acquire 16 unseen frames inside their `text_action` window (`equal_budget`), guaranteeing exact `48`-frame visual parity across all three arms.
   - Combine Focus-1 + Focus-2 text (`combine_observation`) and run unmodified pure-text QA.

## Pre-Registered Endpoints & Single-Unsealing Rule
- **Strict Blindness**: Preflight, GPU inference, and CPU release audit assert `right_answer` and `clue_intervals` decode counts == `0`.
- **Single Unsealing**: Only after `96/96` complete, GPU is verified released (`0 MiB`), and independent CPU `confirm_audit.py release` writes `release.json`, `confirm_score.py` decodes the 96 `q1` labels and clue intervals ONCE.
- **Primary Endpoints (N = 96, exact two-sided McNemar test)**:
  1. `Focus-1+2 native_query (equal_budget, 48 frames)`: `state_split` vs `text` and `state_split` vs `clock_split`.
  2. `Focus-1+2 native_query (on_policy)`: `state_split` vs `text` and `state_split` vs `clock_split`.
- **Secondary Endpoints**:
  1. `Focus-1 native (32 frames)`: `state_split` vs `text` and `state_split` vs `clock_split`.
  2. Official new clue hits (Focus-1 excluding Source frames, and Focus-2 clue hits).
  3. Content control pass rate, `K` distribution, and per-stage wall-clock latency breakdown.
