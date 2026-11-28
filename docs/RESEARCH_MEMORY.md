# 科研记忆｜通信如何改变主动取证的工具参数

项目：Qwen3.5 视频 latent / CG-Bench。更新：2026-11-26（北京时间）。这份文件是跨对话的科研记忆入口，实验原始记录、冻结协议和审计产物为准。96 视频一次性独立盲确认已全量关账。

先读[科研工作与结果速览](RESEARCH_HANDOFF_20261123.md)：按阶段列出做过的实验、得到的证据和可接手的下一步。

## 当前交接

**2026-11-28 21:02 CST｜Stage 2 `Qwen3.5-9B` × `Video-Holmes-315`（`N = 315`, `max_steps = 8`, Option B, 单卡 GPU 0）全量关账并完成 CPU 单次解封评分（`score_qwen35_9b_holmes315_s8.json`: `64d86950...`），零训练反超 `VideoSeek (GPT-5)` 与 `Video-o3-7B`！GPU 0 已自动无缝切入 `Video-MME-306`（`5 / 306`）！**
- **里程碑主结果（`N = 315`, 7×45, `invalid_answers = {}` 零拒答, `959/960 = 99.90%` 控制通过率, `visual_forwards == 0`）**：
  - `vanilla_text`：`126 / 315 (40.00%)`（`5.51` 步 / `4.58` 次视觉调用 / `89.30` 帧；`S4 42.54% → S6 41.90% → S8 40.00%` 明显长链漂移 `−2.54 pp`）；
  - `lave_route_text`：`113 / 315 (35.87%, −4.13 pp)`，但 **大幅削减视觉调用 `−11.6%`（`4.05 vs 4.58`）与 `−8.5` 帧（`80.76 vs 89.30`）**；
  - `vanilla_latent_qa`：`143 / 315 (45.40%, +5.40 pp, p = 0.0857)`；
  - **`lave_full`（+Mech A + Mech B `clean_obs`）**：**`145 / 315 (46.03%)`**（vs `vanilla_text +56/−37, +6.03 pp, p = 0.0614`；胜 `vanilla_latent_qa` `+0.63 pp` 且省 `−11.6%` 视觉调用；`TA` 达 `22/45`）；
  - **`lave_full_visonly`（+Mech A + Mech B 纯视觉状态）**：**`150 / 315 (47.62%)`**（`+64/−40`, **`+7.62 pp, p = 0.0236 *`**，**超越 `VideoSeek (GPT-5) 47.3%`** 且省 **`−11.6%` 视觉调用**）；
  - **`vanilla_latent_visonly`（仅 Mech B 纯视觉状态）**：**`154 / 315 (48.89%)`**（`+67/−39`, **`+8.89 pp, p = 0.00842 ** `**，**超越 `VideoSeek (GPT-5) 47.3%` 达 `+1.59 pp`、超越 573k SFT+RL 训练的 `Video-o3-7B (46.5%)` 达 `+2.39 pp`**，100% Training-Free、0 额外视觉编码！）。

**2026-11-28 07:28 CST｜Stage 1 `Qwen2.5-VL-7B-Instruct` 全部三套 `N >= 300` 主实验（合计 `927` 题，`max_steps = 8`，Option B）+ `VideoSeek-Image (N = 315)` + `n = 60` 无共享端到端延迟审计全部关账，GPU 0 干净归零并无缝切入 Stage 2 `Qwen3.5-9B`（`s = 8`）！**
- **`LongVideoBench-306`（`score_qwen25vl_lvb306_s8.json`: `426bf97a...`, `N = 306`, 17 类, `138 dg=600 + 168 dg=3600`）**：
  - `vanilla_text` `102/306 (33.33%)`（`3.87` 次视觉调用 / `77.86` 帧）→ `lave_route_text` `110/306 (35.95%, +21/−13, +2.61 pp)`（**`3.56` 次 `−8.0%` / `72.94` 帧 / `0` 无效工具调用**）→ `lave_full` `158/306 (51.63%, +78/−22, +18.30 pp, p < 1e-7)` → **`lave_full_visonly 161/306 (52.61%, +88/−29, +19.28 pp, p < 1e-7)`**（17 类 **14 胜 1 平 2 负**；超长 1 小时 `dg=3600 (n=168)` 达 **`51.19%` vs `vanilla_text 30.36% (+20.83 pp, p = 2.2e-5)` 且胜 `vanilla_latent_visonly 50.00% (+1.19 pp)`**！）。
- **Stage 1 三基准总合集（`Qwen2.5-VL-7B-Instruct`, `N = 927`）**：`vanilla_text 315/927 (33.98%)` → `lave_route_text 319 (34.41%)` → `lave_full 436 (47.03%, +13.05 pp)` → `vanilla_latent_visonly 452 (48.76%)` → **`lave_full_visonly 458/927 (49.41%, +233/−90, +15.43 pp, p < 1e-15)`**，同时 **3/3 基准全线减少视觉调用 `−9.2%`（`−9.8% / −9.8% / −8.0%`）**！
- **通宵 GPU 2 Stage 3A 双项关账（`score_qwen25vl_holmes315_img_lat_s8.json`: `9bcad942...`）**：
  - **`VideoSeek-Image (N = 315)`**：`114/315 (36.19%)`（均题反复喂入 `146.1` 帧 / `30,713` prompt tokens、被迫裁剪 `33.4` 帧），**`lave_full_visonly (38.10%)` 反超 `+1.90 pp (+48/−42)`** 且零回放、零裁剪；
  - **`n = 60` 无共享端到端延迟审计（Table 6b）**：`vanilla_text 49.35 s` vs `lave_route_text 52.24 s (+5.9%, rx 3.90s)` vs **`lave_full_visonly 55.07 s (+5.72 s / +11.6%, 纯视觉读出仅 0.27 s)`**。
- **晨间切换状态**：GPU 0 已无缝启动 Stage 2 `Qwen3.5-9B`（`holmes315, mme306, lvb306`, `s=8`, `mamba_cache_mode='align'`）；GPU 2 当前正在跑 Table 2 五臂消融 `traj_bundle (68/315)` 并将在 `07:45 CST` 自动归零还卡。

**2026-11-27 23:18 CST｜`Qwen2.5-VL-7B-Instruct` 单卡（仅 GPU 0）`Video-MME-306`（`N = 306`, `102S/102M/102L`, 12 任务, `max_steps = 8`, Option B）全量关账并完成 CPU 单次解封评分（`score_qwen25vl_mme306_s8.json`: `4c389486...`），GPU 0 已自动无缝切入 `LongVideoBench-306`（已跑至 `10/306`）！**
- **里程碑主结果（`N = 306` 独立视频，`vanilla_invalid_tool_steps = 0` & `lave_invalid_tool_steps = 0` 双臂零无效工具调用，`494/513 = 96.30%` 控制通过率，`visual_forwards == 0` 严格断言）**：
  - `vanilla_text`：`132 / 306 (43.14%)`（S `50.98%` / M `44.12%` / L `34.31%`；`3.93` 步 / `2.97` 次视觉调用 / `63.58` 帧）；
  - `lave_route_text`：`123 / 306 (40.20%, −2.94 pp)`（纯文本回答受 7B 字幕压缩瓶颈拖累），但 **同步减少视觉调用 `−9.8%`（`2.68 vs 2.97`）与 `−4.76` 帧（`58.82 vs 63.58`）**，Receiver 仅 `2.70 s/case`；
  - `vanilla_latent_qa`：`169 / 306 (55.23%, +12.09 pp, p = 0.000341)`；
  - **`lave_full`（+Mech A + Mech B 过滤空拒答 `clean_obs`）**：**`174 / 306 (56.86%)`** —— 对 `vanilla_text` **`+75/−33 (+13.73 pp, p = 0.000066)`**，对 `vanilla_latent_qa` **`+28/−23 (+1.63 pp)`**（改道子集 `+3.19 pp`），**Short (`73.53%`, `+22.55 pp`) / Medium (`57.84%`, `+13.73 pp`) / Long (`39.22%`, `+4.90 pp`) 全三档零回退**；
  - `vanilla_latent_visonly`：`173 / 306 (56.54%, +13.40 pp, p = 0.000125)`；
  - **`lave_full_visonly`（+Mech A 路由 + Mech B 纯视觉状态 Top-8 读出）**：**`177 / 306 (57.84%)`** —— 对 `vanilla_text` **`+77/−32 (+14.71 pp, 精确双侧 McNemar p = 0.000019 < 2e-5)`**，对 `vanilla_latent_visonly` **`+23/−19 (+1.31 pp)`**（`n = 188` 改道子集 `55.32% vs 53.19%, +2.13 pp`），12 大题型中 **10 胜 2 负**（Short `77.45%`、Medium `61.76%` 大幅领先；Long 上保留非空时间线文本的 `lave_full` 达 `39.22%`，较 `visonly` 高 `+4.90 pp`，揭示超长视频需保留全局时间线锚点）。
- **两套 `N >= 300` 已关账合集（`Qwen2.5-VL-7B`, `N = 621`）**：`vanilla_text 213/621 (34.30%)` → **`lave_full_visonly 297/621 (47.83%, +13.53 pp)`**，均节省 **`−9.8%`** 视觉调用！

**2026-11-27 16:31 CST｜`Qwen2.5-VL-7B-Instruct` 单卡（仅 GPU 0）`Video-Holmes-315`（`N = 315`, `max_steps = 8`, Option B 纯视觉状态读出 `visonly`）全量关账并完成 CPU 单次解封评分（`score_qwen25vl_holmes315_s8.json`: `34499116...`），同步推进 `Video-MME-306`（GPU 1/2 全程 `0 MiB`）！**
- **里程碑主结果（`N = 315`, 7 类 × 45 题, 覆盖 Video-Holmes 全部 `270/270` 独立视频, `invalid_answers = {}` 零拒答, `vanilla_invalid_tool_steps = 0`, Receiver & Mech B 严格断言 `visual_forwards == 0`）**：
  - `vanilla_text`（8-step Caption VideoSeek）：`81 / 315 (25.71%)`（`4.71` 步 / `3.79` 次视觉调用 / `76.70` 帧）；
  - `lave_route_text`（+Mech A 视觉状态路由）：`86 / 315 (27.30%)`（`+20/−15`, `+1.59 pp`），**同步省步数与视觉调用（`4.39` 步 / `3.42` 次 `−9.8%` / `70.76` 帧 `−5.9f`，Receiver 仅 `3.81 s/case`，控制通过率 `739/763 = 96.85%`）**；
  - `lave_full`（+Mech A + Mech B 过滤空拒答 `clean_obs`）：`104 / 315 (33.02%)`（`+50/−27`, **`+7.30 pp`, `p = 0.01167 *`**）；
  - `vanilla_latent_qa`（+Mech B `clean_obs`）：`107 / 315 (33.97%)`（`+55/−29`, **`+8.25 pp`, `p = 0.00604 ** `**）；
  - `vanilla_latent_visonly`（+Mech B 纯视觉状态）：`114 / 315 (36.19%)`（`+62/−29`, **`+10.48 pp`, `p = 0.000705 *** `**）；
  - **`lave_full_visonly`（+Mech A 路由 + Mech B 纯视觉状态 Top-8 读出）**：**`120 / 315 (38.10%)`** —— 对 `vanilla_text` **`+68 赢 / −29 输`（`+12.38 pp`, 双侧精确 McNemar `p = 0.000093 < 1e-4`）**，对 `vanilla_latent_visonly` **`+22 / −16`（`+1.90 pp`）** 呈现清晰 Mech A + Mech B 协同净增益；**7/7 题型全胜 `vanilla_text`**（`MHR +12`, `SR +12`, `TCI +7`, `CTI +3`, `IMC +3`, `PAR +1`, `TA +1`），在 `n = 246` 改道子集达 **`39.43% vs 27.24%`（`+12.19 pp`）**，100% Training-Free 超越已发表同基座 RL 模型 `VideoChat-R1-7B (33.0%)`、`SEED-Bench-R1-7B (33.5%)` 与 `Video-R1-7B (36.5%)`！
- **单卡续跑状态**：A40 GPU 0（`PID 3905127`, `run_qwen25vl_main --gpu 0 --max-steps 8 --suites holmes315,mme306,lvb306 --refresh-mech-b`）已自动切入 `mme306_s8`（`N = 306`），随后自动跑 `lvb306_s8`（`N = 306`）；GPU 1 & 2 严格保持 `0 MiB`。

**2026-11-26 晚至 2026-11-27 早三卡 A40 通宵实验全量关账，全部占用 GPU（GPU 0/1/2）已于 07:45:39 CST 准时自动归零释放（`0 MiB`, `0%`，提前 14 分钟归还，全程未触碰 GPU 3），所有封存评分回执已同步至 `wangyam1/LAVE2027`。**
- **Video-MME 216 多步评测（`score_mme.json`: `df6f2fad...`）**：216 独立视频（Short 72 / Medium 72 / Long 72，12 题型均衡），`vanilla_text 125/216 (57.87%)` → `lave_route_text 131 (60.65%, +2.78 pp)` → **`lave_full 138 (63.89%, +26/−13, +6.02 pp, p = 0.053)`** → **`vanilla_latent_qa 141 (65.28%, +23/−7, +7.41 pp, p = 0.00522)`**；Short/Medium/Long **全三档零回退**，同时 LAVE 减少视觉调用 **`−8.7%`**（`3.04 vs 3.33`）与采样帧数 `64.67 vs 69.26`（`440/441` 控制通过）。
- **LongVideoBench-150 多步评测（`score_lvb150_multistep.json`: `cf8d73d0...`）**：沿用冻结 `PIN 27b9d833...`（150 题 / 146 独立长视频 / 17 类），`lave_route_text` **`87/150 (58.00%)`** vs `vanilla_text 85/150 (56.67%)`（`+1.33 pp`），其中 **超长 1 小时分桶（`dg=3600`, `n=101`）大涨 `+5.94 pp`（`59.41% vs 53.47%`）**；均题视觉调用 `−6.0%`（`4.73 vs 5.03`）、帧数 `−4.8`（`91.63 vs 96.43`）。三大多步基准合并（`N = 506`）：`vanilla_text 49.41%` → `lave_route_text 52.96% (+3.56 pp)` / `lave_full 53.36% (+3.95 pp)`，且 3/3 基准全部同步省步数、省调用、省帧数。
- **Video-Holmes eval140 审稿人反杀基线与独立无共享延迟审计（`score_baselines_eval140.json`: `a869553e...`，正面回应 LARVA ICLR 2027 撤稿审稿人核心质疑）**：
  - **`VideoSeek-Image`（直传原始像素帧多轮交错）**：`40/140 (28.57%)`（与 `vanilla_text` 持平 `+0.00 pp`，累计喂入 `216.7` 帧 / `44,953` prompt tokens 且被迫裁剪 `49.1` 帧）——`lave_route_text (35.71%)` 与 `lave_full (34.29%)` 分别领先 **`+7.14 pp` / `+5.71 pp`**；
  - **`Clock-Split`（等预算 50/50 几何切分）**：`48/140 (34.29%)`、`clock_split_uniform_qa 46/140 (32.86%)`——LAVE 在 Mech A（`35.71% vs 34.29%`）与 Mech B（`34.29% vs 32.86%`）双双胜出且更省调用和帧数；
  - **28 题无共享端到端延迟审计**：`lave_route_text` 实测 **`70.81 s`**，**直接快于 `vanilla_text`（`70.95 s`）**——`7.76 s` 零像素探测开销被 **`−18.3%` 视觉调用（`3.18 vs 3.89`）** 与 **`−14.6%` 帧数（`66.86 vs 78.29`）** 完全抵消！
- **多规模 / 跨架构 / Thinking 模式泛化（全在同套冻结 `Video-Holmes eval140` 上验证，零魔改原生 vLLM 0.19.0）**：
  - **`Qwen3.5-4B`（`score_qwen35_4b.json`: `482060fa...`, `N=140`）**：`vanilla_text 25.00%` → `lave_route_text 25.71%` → `lave_full 27.86% (+2.86 pp)` → **`vanilla_latent_qa 52/140 (37.14%, +22/−5, +12.14 pp, p = 0.00151)`**（4B 零像素读出反超 9B 纯文本 VideoSeek `28.57%` 达 `+8.57 pp`！）；
  - **`Qwen2.5-VL-7B-Instruct`（`score_qwen25vl_7b.json`: `e6593786...`, `N=140`, 纯 Transformer 架构 + LARVA 同基座）**：`vanilla_text 15.71%` → `lave_route_text 17.14% (+1.43 pp)` → **`lave_full 40/140 (28.57%, +24/−6, +12.86 pp, p = 0.00143)`**，视觉调用 `−7.8%`，Receiver 仅 **`1.29 s/case`**、0 视觉回放；
  - **`Qwen3.5-9B-Thinking`（`score_qwen35_9b_thinking.json`: `88a8f385...`, `N=46` 均衡前缀，07:43 准时硬停）**：严格单调四臂阶梯 `23.91% → 26.09% → 30.43% → 32.61%`（`lave_full vs vanilla_text +6/−2, +8.70 pp`; `vs v_lat +3/−0, +6.52 pp`），视觉调用 `−7.3%`、`−5.9` 帧、`164/164` 控制全过。

**Video-Holmes 140 独立长视频多步 VideoSeek + LAVE 全量对齐评测（`videoseek_lave_20261126`，`eval140` + `pilot14`）已在单卡 A40 GPU 1 上全量跑完、释放显存至 `0 MiB`（`gpu_released: true`），完成 CPU 单次解封评分（`score_eval140.json`: `276bb198...`）并推送至 Private GitHub 仓库 `wangyam1/LAVE2027`。**
- **设置与效率**：`eval140` 覆盖 **140 题 / 140 个独立视频**（7 类严格每类 20 题，与 `pilot14` 零重叠），采用官方多步 VideoSeek 协议（`overview / skim / focus / answer`，`max_steps = 6`）、100% Training-Free 冻结 `Qwen3.5-9B` + 原生免魔改 vLLM 0.19.0、**纯单视频三项自检（零跨视频 foil，通过率 `385/385 = 100%`）**，Receiver 与 Top-8 零像素隐状态核验均严格断言 `visual_forwards == 0`。均题步数/视觉调用/帧数：`Vanilla 4.66 步 / 3.86 次 / 77.71 帧` → **`LAVE 4.64 步 / 3.75 次 / 76.00 帧`**（Receiver 均耗时 `10.26s`）；`pilot14`（14 题）上 `lave_route_text` 在持平准确率 `8/14 (57.1%)` 下减少 **`−22.6%` 视觉调用、`−17.5%` 帧数**（`72.0 → 59.4` 帧）。
- **全量 140 题准确率（`max_steps = 6`）**：
  - **`lave_route_text`（机制 A：LAVE 视觉状态路由）**：**`50 / 140`（`35.71%`）** vs `vanilla_text` `40 / 140`（`28.57%`）——**`+18 赢 / −8 输`（净 `+10`, `+7.14 pp`，双侧精确 McNemar `p = 0.0755` / 单侧 `p = 0.0378`；115 题改道子集 `40.0% vs 31.3%`, `+8.70 pp`）**，**7 大推理题型中 6 类严格胜出**（`MHR` 11 vs 8、`IMC` 9 vs 7、`PAR` 8 vs 6、`TCI` 6 vs 4、`CTI` 5 vs 4、`TA` 5 vs 4，仅 `SR` 6 vs 7）；
  - **`lave_full`（机制 A + Top-8 零像素核验）**：**`48 / 140`（`34.29%`）**（vs `vanilla_text` `+21/−13`, `+5.71 pp`；vs `vanilla_latent_qa` `+14/−10`, `+2.86 pp`；`SR` 上冲至 `9/20 = 45.0%`）；
  - **`vanilla_latent_qa`（仅机制 B，0 额外视觉前向）**：**`44 / 140`（`31.43%`）**（vs `vanilla_text` `+12/−8`, `+2.86 pp`）；
  - **多步漂移诊断（Step-4 断点诚实记录）**：`vanilla_step4 = 52/140 (37.14%)` 在后两步盲找干扰下掉点 **`−8.57 pp`** 至 `28.57%`，而 `lave_step4 = 49/140 (35.00%)` 稳步升至 `35.71%`（`+0.71 pp`），印证视觉状态路由显著抑制多步长链漂移。 [eval140 评分](videoseek_lave_20261126/score_eval140.json) · [GitHub 仓库](https://github.com/wangyam1/LAVE2027)

**LongVideoBench 150 类别均衡全量验证（`lvb_vllm_20261126`）已在双卡 A40（GPU 0 + GPU 1）上全量跑完并双卡归零释放（`0 MiB`, `gpu_released: true`），完成 CPU 单次解封评分（`score.json`: `e5c5cfe1...`）。**
- **设置与效率**：150 题覆盖 146 个独立长视频（`duration >= 180s`：`duration_group=3600` 101 题、`600` 49 题）、均匀覆盖全部 17 个 LVB 题型（每类 8–9 题）；运行在隔离环境 `/data1/llj/lljmas/envs/lave_vllm_20261126` 的原生免魔改 **vLLM 0.19.0** 引擎（冻结 `Qwen3.5-9B`，100% Training-Free，`mm_processor_cache_gb=0` 修复默认 4 GiB 缓存驱逐不同步）。控制通过率 **`108 / 130`（`83.1%`）**（与 CG-Bench 96 盲确认的 `69/83 = 83.1%` 精准一致；`K=4`: 101, `K=2`: 5, `K=1`: 2）；均题三臂总耗时 **`70.28 s`**（LAVE Receiver 仅 **`5.07 s`**），Focus-1 均 **`1.91` 轮 / `60.71` 帧**，Focus-1+2 等预算均 **`2.37` 轮 / `67.96` 帧**。
- **全量 150 题准确率**：
  - **`direct_visual_qa_f2`（`on_policy` 与 `equal_budget` 完全一致）**：`state_split` **`87/150`（`58.0%`）** vs `clock_split` `85/150`（`56.7%`）vs `text` `83/150`（`55.3%`）——`state_split` vs `text` **`+5 赢 / −1 输`（净 `+4`, `+2.67 pp`, 双侧精确 McNemar `p = 0.219`；在 108 改道题上 `65.7% vs 62.0%`, `+3.70 pp`）**，`state_split` vs `clock_split` **`+4 / −2`（`+1.33 pp`, `p = 0.688`）**；17 个题型上 **4 胜 13 平 0 负（零退化）**；`dg=3600`（`n=101`）为 `57/57/54`（`+4/−1`），`dg=600`（`n=49`）为 **`30/28/29`**（vs `clock` `+2/−0`，vs `text` `+1/−0`）。
  - **`direct_visual_qa_f1`**：`state` `85 (56.7%)` / `text` `84 (56.0%)` / `clock` `84 (56.0%)`（`+3/−2`）。
  - **纯文本 QA（再次印证 Focus→文本压缩瓶颈）**：`focus1_native` `clock 82 / state 81 / text 80`，`focus12` `state 81 / clock 81 / text 80`（108 次改道中纯文本 `qa1` 仅 `4/108` 改选字母；移除文本瓶颈的 `direct_visual_qa_f2` 叠加 LAVE 路由，相对纯文本 `text` 基线 `80/150 = 53.3%` 总计净增 **`+4.67 pp` 至 `58.0%`**）。 [LVB-150 完整记录](lvb_vllm_20261126/notes.md) · [LVB-150 评分](lvb_vllm_20261126/score.json) · [冻结清单](lvb_vllm_20261126/preflight.json)

**96 个未曝光 CG-Bench 长视频一次性独立盲确认（`confirm96_20261126`）已在单卡独占 A40 GPU 0 全量跑完、释放显存至 `0 MiB`、通过无标签独立 CPU 审计（`release.json`: `f8366fd6...`，峰值 19.95 GiB，0 无效 QA）并完成一次性解封计分（`score.json`: `4fc650f0...`）。**
- **强正向确认（主动线索定位泛化）**：83 个可扫描视频中 **69/83（83.1%）** 通过全部三项严格内容控制（`K=4`: 65，`K=2`: 4）。排除 Source 已见帧的 Focus-1 新线索命中为 **`state_split` 41/96（42.7%）vs `clock_split` 36/96（37.5%）vs `text` 26/96（27.1%）**——`state_split` 对 `text` **`+17 赢 / −2 输`（净增 `+15` / `+15.6 pp`，双侧精确 McNemar `p = 0.000729`；在 69 触发题上 `52.2% vs 30.4%`，`+21.7 pp`）**，对等预算 `clock_split` 为 **`+11 赢 / −6 输`（`p = 0.332`）**；Focus-2 线索命中同样由 `state_split` 领先（`on_policy` **16 vs 13 vs 11**；等预算总计 **20 vs 19 vs 13**）；Focus-1 空报告（`No relevant content found.`）在 69 触发题上由 `29/69` 降至 `20/69`（−31%）。
- **诚实零结果（端到端纯文字 QA 转化未确认）**：Focus-1 原生纯文字 QA 为 `text` **34/96（35.4%）**、`state_split` **33/96（34.4%）**、`clock_split` **33/96（34.4%）**（`state_split` vs `text`：`+0/−1`，`p = 1.0`）；Focus-1+2（`on_policy` 与 `equal_budget` 逐题一致）为 `clock_split` **34/96（35.4%）**、`text` **33/96（34.4%）**、`state_split` **32/96（33.3%）**（vs `text`：`+1/−2`, `p = 1.0`；vs `clock_split`：`+1/−3`, `p = 0.625`）。
- **解封后只读根因诊断**：在 69 个 Focus-1 帧完全不同的确认题上，**`67/69`（97.1%）在 QA-1 选了完全相同的选项字母**（QA-2 为 `64/69`），原因集中在三点：(1) 未修改的原生 Focus-1 提示词在多例上复读查询而不输出具体视觉属性（如色号、物体特征）；(2) 长视频上 `K=4`（每缝 4 帧）使 11/17 独有线索命中仅含 1 帧，且 Focus-2 聚类偏向 16 帧旧文字锚点而非新发现的状态缝隙；(3) 纯文字 QA 强锚定于前两份长 Source 报告。 [确认完整记录](confirm96_20261126/notes.md) · [确认审计](confirm96_20261126/release.json) · [确认评分](confirm96_20261126/score.json)
- **工程加速实测（`vllm_lave_20261126`，已释放 GPU 至 `0 MiB`）**：在只读克隆 `toolrl2` 的隔离环境 `/data1/llj/lljmas/envs/lave_vllm_20261126`（`vllm 0.19.0` + `flash_attn 2.8.3` + `fla 0.4.2`，零改动原环境与 vLLM 源码）上，通过 `Qwen3_5ForConditionalGeneration` 原生 `enable_mm_embeds=True`、`image_embeds` + M-RoPE、`logprobs_mode="processed_logits"`、`mamba_cache_mode="align"` 前缀缓存及 `LLM.apply_model` 视觉禁用守卫，在 4 例开发探针（`case_16/06/20/38`）实测：**Receiver 全量（33 槽 + 3 项对照）由 HF `14.60 s` 降至 `4.36 s`（实测 `3.35×` 提速，严格 0 视觉前向）**，`source_shared` 由 `29.99 s` 降至 `20.78 s`，`Focus-1` 由 `8.82 s` 降至 `7.57 s`；视觉网格 `4/4` 全一致、三项控制 `4/4` 全过、`zero_margin_diff = 0.000–0.125`，`case_16` 选出完全相同 4 缝集合 `{0,4,6,8}`、`case_06` Top-3 `[2,7,27]` 一致（BF16 算子均位 `lift` 差 `0.15–0.27`）。 [原型引擎](vllm_lave_20261126/engine.py) · [基准回执](vllm_lave_20261126/bench_vllm_b33.json)
- **Focus→QA 转化瓶颈消融与视觉上限诊断（`focus2_synth_20261126`，51 开发视频，已释放 GPU 至 `0 MiB`，不触碰 96 确认集）**：针对 96 确认集暴露出的原生 Focus 逐帧复读查询退化（如 `case_05/08/17/00` 连续复读 8–16 行查询而不报数量），在完全相同的 16 张 Focus-2 封存原图上对照三路读出（峰值 19.29 GiB）：
  - 原生 Focus-2 基线：`19 / 14 / 15`（`on_policy`）、`19 / 15 / 16`（`equal_budget`）；
  - 选项盲跨帧综合 Focus-2 → 未改纯文字 QA（`synth_qa_native`）：`19 / 16 / 16`（`on_policy`，`+3/−0` vs `text`）、`19 / 17 / 19`（`equal_budget`）——成功救回 `case_17`（茶叶蛋 `F→D`）与 `case_48`（`E→H`），但自由描述在 `case_06`（出现幻觉人物名与时间）和 `case_38` 翻输两例，总分持平 `19`；
  - 加特写优先提示的纯文字 QA（`synth_qa_priority`）：`17 / 15 / 16`（`on_policy`），放大 Focus-2 幻觉反而掉 2 分；
  - **直接视觉 QA 上限诊断（`direct_visual_qa`，直接看完全相同的 16 张 Focus-2 帧 + 选项 `A–H`）**：`18 / 16 / 14`（`on_policy`）、`18 / 18 / 16`（`equal_budget`）——证明即使彻底移除文本压缩瓶颈，冻结 `Qwen3.5-9B` 在 `262,144` 像素/帧分辨率下的细粒度计数/小字幕视觉感知本身已达 ~`18–19/51`（~35–37%）天花板，并非单靠改 Focus/QA 提示词就能突破。 [消融脚本](focus2_synth_20261126/synth_run.py) · [开发集评分](focus2_synth_20261126/score.json)

**此前已曝光 51 开发视频记录（LAVE 熵控多区间路由 + Focus-2 密集聚焦核验）：**
- **阶段一（Focus-1 熵控多区间路由，`entropy_routing_20261125`）**：去掉未校准 `margin > 0` 先验门槛并按论文 Introduction 引入归一化香农熵三分位 NMS 多区间分配（固定 16+16=32 帧，`K ∈ {1, 2, 4}`，匹配等 K 贪心最远点时钟）。42/44 可扫描视频通过全部三项严格内容对照（`shifted` / `scrambled` / `wrong_target`）；排除 Source 已见帧的官方开发集新线索命中为 **`state_split` 21 / `text` 15 / `clock_split` 14**（对 `clock_split` **+8 赢 / −1 输，双侧精确 McNemar p = 0.0391**，对 `text` **+8 赢 / −2 输**）；未修改原生报告纯文字 QA 为 **`17 / 14 / 15`**（对 `text` **+3 赢 / 0 输**，对 `clock_split` **+2 赢 / 0 输**）。 [阶段一审计](entropy_routing_20261125/release.json) · [阶段一评分](entropy_routing_20261125/score.json)
- **阶段二（Focus-2 16 帧密集聚焦核验，`zoom_verify_20261125`）**：按各臂自身 Focus-1 原生报告非空时间戳的最密 30 秒聚类（±10 秒）采 16 张未见密集帧，使用 100% 未修改原生 Focus 模板与原 Tool query（`native_query`），在 27 个触发题上同时报告自然条件触发（`on_policy`）与严格等视觉预算（`equal_budget`，三臂严格同为 48 帧 / 12,096 视觉 token）。Focus-2 线索命中为 `15 / 11 / 6`（`on_policy`，对时钟 **+9 赢 / 0 输，p = 0.0039**）；端到端纯文字 QA 进一步升至 **`19 / 14 / 15`**（`on_policy`：对 `text` **+5 赢 / 0 输，+9.8 pp，双侧 p = 0.0625 / 单侧 p = 0.03125**，对 `clock_split` **+4 赢 / 0 输**）与 **`19 / 15 / 16`**（严格等预算 48 帧：对 `text` **+4 赢 / 0 输**，对 `clock_split` **+3 赢 / 0 输**），零输例（含 `case_16` 金豪 80 读出 `292.7s: 金豪的80大概20多块钱` 翻对、`case_06`、`case_20`、`case_38`、`case_41`）。96 个独立确认视频继续完全封存，等待确认是否冻结独立协议做一次性确认验证。 [完整记录](entropy_routing_20261125/notes.md) · [阶段二审计](entropy_routing_20261125/zoom_verify.release.json) · [阶段二评分](entropy_routing_20261125/zoom_verify.score.json)

- **协议冻结前两项核验（均在 51 开发视频内完成，GPU 已释放，96 确认集未触碰）**：(1) 5 个胜例实拍帧视觉复核确认 `case_06`（`223.9s` 绿卡车撞碎悬空石且无自行车）与 `case_16`（`292.7s` 字幕「金豪的80大概20多块钱」）为严格 LAVE 独有新帧因果胜例，其余 3 例（`case_20/38/41`）画面真实但共享相近区域或锚点，0 例幻觉；(2) A40 实测 `batch_size=1/4/8/11` 接收器前向显示 `batch=1` 平均仅 **14.6 秒/题**（含全部 3 项对照）、峰值 **18.2 GiB**、**0 漂移**，在长提示下反而快于且稳于 `batch=4/8/11`（后者在 `case_20` 引入 0.25 BF16 漂移且峰值达 24.1 GiB），故 96 视频确认协议应冻结 `batch_size=1`。 [批处理基准](entropy_routing_20261125/bench_batched.json)

**`UNCERTAIN` 补帧与 F1 状态选边两轮对照均已在已曝光 51 视频上完成单卡运行、CPU 审计并释放 GPU。** 补帧对照（26 份可配对 `UNCERTAIN` 报告，各补 8 张新帧）核验后 QA 为：引用附近 `16/14/16`、原区间分散 `17/16/17`（次序 `state_split / text / clock_split`，均 `/51`）。F1 状态选边对照（10 份满足 ±2 秒等预算前后窗条件的报告，共 20 次 8 帧读出）显示：F1 精确复现 8/10，纯文字 Planner 10/10 全选 `AFTER`，F1 状态通过全部内容对照 3/8（仅 `case_04` 大葱题改选 `before` 区别于文字）；受限于 13/20 次 F2 报告因引用 >2 帧被判 `invalid`，未核验 QA 三策略均较基线降 1（`15/14/15`），核验后三策略均持平基线（`16/15/16`，彼此 0 赢 0 输）。A40 1 号卡已释放至 0 MiB，96 个确认视频继续封存。 [补帧评分](missing_fact_routing_20261124/refocus.score.json) · [选边评分](missing_fact_routing_20261124/focus_side.score.json)

**缺失事实改版及引用帧核验均已在已曝光 51 视频上关账。** 11 月 24 日的原报告 QA 是 latent／文字／时钟 **14／13／13**，定向报告是 **16／15／16**，latent 对时钟 1 赢 1 输。11 月 25 日在 A40 1 号卡完成 56 份报告的固定帧核验并释放。相同提示使用报告引用帧得到 26 次支持、等预算远端帧得到 5 次支持，配对 22 赢 1 输。仅递送被支持报告后，三组 QA 降为 **14／12／13**。净损失来自把不确定当作弃答，仅拦截明确拒绝则逐题正误不变。钢笔题保留正确价格，大葱题丢弃原本答对的计数报告。96 个确认视频封存。 [运行台账](missing_fact_routing_20261124/task_plan.md) · [两轮结果](missing_fact_routing_20261124/notes.md)

**LAVE 的缺失事实改版已接入单题流程。** 钢笔价格胜例的新增画面拍到目标字幕，旧计数题的线索帧错过举起大葱的瞬间。新方案沿用原生 Tool query，从原题另行提取待查的实体、属性和时间条件。接收器将这个目标用于视觉状态选址，Focus 在采到的帧里尝试报告该事实并给出帧引用。三组选窗与两种报告形成同题配对，最终复用原生纯文字 QA。通过合成病例、原生渲染配对和微型随机权重 Qwen3.5 接收器的 CPU 验收。 [本地验收](missing_fact_routing_20261124/task_plan.md) · [判读规则](missing_fact_routing_20261124/notes.md)

**已完成旧实验复现、单槽方案关账和 16+16 候选的实际采帧；新候选的两阶段盲审和 QA 已关账。** 原单槽整窗方案最终 QA **11/51 vs 12/51**（对强文字，0 赢 1 输）。它采到原报告没写的新可见事实 **17/51 vs 13/51**（6 赢 2 输），但两组经核验写入 Focus 报告的新增事实均为 **0**。本轮定位到了“看见 → 写进报告”的断点；固定查询不匹配和报告提示词促使弃答，这两个解释还没有分开。同帧报告实验已启动，固定历史 query。

**改进与证据边界。** 16+16 保留一半强文字选窗，用另一半验证 latent 能否选出额外证据。官方线索覆盖为 **18/51（latent 加窗）／15/51（全量文字）／15/51（时钟加窗）**，对两个对照各 **4 赢、1 输**；这是开发集上的定位信号。三组已完成 **92/92** 次实际取帧，零结构审计失败，时钟组有一次生成触顶；同题视觉预算对齐，A40 作业已释放 GPU。47 个合法视频、92 个视图已导出并验收为 19 个盲包；23:43 已签收全部 **19/19**（47 个视频、92 个视图）。6 个匿名报告包、56 条事实全部通过独立复核。三组最终 QA 各 12/51，逐题正确性相同。状态窗的新可见事实为 12/51，文字 9/51、时钟 8/51；未截断报告中实际递送分别为 0/51、0/51、1/51。见[开发集关账](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/efficacy_readout_20261123.md)。

继续工作请先看[候选续跑检查点](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/resume_checkpoint.md)和[活动实验台账](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/task_plan.md)。这两个入口分别记录下一项实际动作和原始进度；本文件保留研究脉络与证据索引。

## 研究问题和目前判断

原 Introduction 与 16+16 实际运行的区别见[实证对照](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/introduction_fidelity.md)。关键限制：Planner 没有重看像素，但审计用的单槽状态来自原 Source 的重放 forward；[现场捕获接入验收](live_source_preflight/README.md)先后在微型 CPU 和真实 9B Tool 上通过：两帧合成输入的一次视觉编码捕获了 162 个视觉 token，状态与测量用重放逐元素一致，12 个生成 token 不变。在首个冻结触发视频的原 32 帧上，一次编码捕获的全部 32 帧指纹与封存重放相同；固定历史接收上下文后选中同一地址。报告和接收分数未精确重现：报告同为 727 token，但前缀仅 100 token 一致，另一次同输入带捕获生成了 798 token。不能据此判断漂移是不是捕获引入。[同会话开关对照](live_source_preflight/source_drift_probe.receipt.json)已完成：同一已曝光视频按「关→开→开→关」实际生成四次，四次报告的 727 token 与历史逐 token 一致，开启两次各 32/32 帧状态指纹一致；先前报告漂移在此次没有复现，原因未确定。[两份实时报告进入 Planner](live_source_preflight/live_planner_probe.receipt.json)也已在同一已曝光视频实测：Planner 不接收像素，生成有效且不同于旧档的文字工具参数；同一运行的状态地址与历史一致、零额外 Source 重放。[进一步已完成三组现场取帧](live_source_preflight/live_focus_probe.receipt.json)：三组各 32 帧、8,064 视觉 token、无截断。先由实际报告和现场状态分别生成新参数及地址，再用当次参数执行，三套帧下标互异。控制选址共 64 次前向、约 30.27 秒；单例不能证明有效或更快。

Tool 完成一次视觉观察和 prefill 后，视觉状态里是否保留了完整文字报告没表达、且能改变下一步取证决策的信息？我们研究的是**发送端实际视觉状态 → 接收端对内容的选择 → 真实工具时间参数 → 新的可见事实 → 文字报告 → 独立 text-only QA**。显式参数是 `start_time/end_time`；当前候选固定原文字方案的 `query`，不把自由生成整组参数说成已经解决。角色由预训练模型实现；这轮验证不等于跨不同模型家族传输。

**目前能确认**：早期 latent 消息会改变动作；冻结的单槽读出在部分视频上对内容干预有反应；保留文字窗口的 16+16 候选提高了已曝光开发集上的官方线索覆盖；三组实际取帧的帧数和视觉 token 已经逐题对齐。**目前不能确认**：16+16 候选能递送更多有用事实、最终提分、节省端到端时间，或能在未曝光视频泛化。几何命中、选择分数和报告输出都不是答题效果。

约束：完全无训练；不沿用 HyFunc，也不把它当作支持证据。早期有训练桥的历史只作负结果归档；当前方案及其后续检验均完全无训练。需要 GPU 时只用一张卡，本轮用 A40 1 号卡，25 GiB PyTorch 上限，跑完即释放；盲审用本地/CPU，不占卡。发送端不额外向 Planner 重放视频是原 Introduction 的机制要求；普通真图重看只是内容是否有用的诊断。BFCL/tau 的工具调用动机保留为背景，这份 CG-Bench 证据不能外推为 BFCL/tau 的结果。不要把旧 Introduction 的 62.8%、+3.8/+1.2、6.2 秒当成本轮已复现的数字。

## 已完成的工作和保留的负结果

| 阶段 | 我做了什么、结果说明什么 |
| --- | --- |
| 早期工具参数通信（独立的开发线） | 将问题从“向模型传软消息”收窄为“接收器能否用消息内容改对一个具体工具参数”。历史 BFCL/xLAM 做了五种子参数寻址、文字 draft 和同预算随机寻址的对照。被寻址方案较 draft 在 BFCL 开发切片低 **0.94 个百分点**，xLAM 持平，未通过门槛；该实验没有跑 tau-bench。只保留为历史负结果；当前视频线完全无训练，也不从 HyFunc 借用方法或正面证据。 |
| 环境镜像与迁移 | 交付了 `pytorch` 类型的 Qwen3.5 软件环境 `.tar.gz`，用独立构建目录保留 Dockerfile 和 `.dockerignore`，验证归档、Jupyter、SSH 与小模型 GPU 路径。按要求镜像只含环境，不打包实验代码、权重、数据或凭据；A100 复现是另外部署并运行的。 |
| A40 → A100 环境迁移与旧 101 题复现 | 验证了环境、权重、多模态推理和日志；冻结的 A100 对照只跑 latent/no-latent，分别 **60/101、59/101**。另核对 A40 原档四组：真实 **67/101**、空消息 **61/101**、值置换 **68/101**、时间置换 **60/101**。真实消息对无消息改变首个可比工具动作 **47/100**，但值置换也能改变动作且准确率不低于真实消息。结论：有行为影响，真实消息的内容优势没确立。用户指定的 66% 是[工作比较线](../qwen35_video_improvements_20261110/README.md)，A40 历史实测为 67/101（66.34%），并非 A100 的实测。A40 和 A100 是不同运行，现有证据不能把差异归因于 GPU。 |
| 机制图与 PCA | 画了共享骨干角色通信图、逐题配对得失和路由诊断，定位到熵规则几乎总选满 8 组、几乎所有被选组实际只有 1 个 token。PCA 使用题级日志中的路由行为特征；没有存下可做原始 latent PCA 的高维向量，也没看到获益与损伤题清晰分开。 |
| 接收器与 8 视频先导 | 核对过残差注入、缓存/特征读出、注入时机、注意力时间地址，以及空、噪声、错问题/时间等对照。H19 冻结读出真实改写过状态，但 8/8 的 real/zero 完整参数不变。另一次真实图像先导 156 次调用：真实图像 **3/8**，两种文字基线各 **4/8**；逐题 **0 赢、1 输**。这证明固定设计未过门槛，不能说所有无训练接口都无效。小样本不用来宣称可复现增益。原 8 题人工草稿按用户确认由未看过本轮答案或模型输出的同学填写，6 处留白均为已审但无法回答；两名独立 AI 对剩余观察作[补充复核](../deliverables/video_fact_review_ai_supplement_20261116/AI补充复核.md)，补充与原人工记录分开保留。 |
| 早期离散地址和 GT 线索诊断 | 历史独立参数主集 15 视频：离散窗口 **8/15**，连续消息 **1/15**（7 赢、0 输），同源文字和直接规则各 **11/15**。真实 Focus 下游三组（离散、连续、无消息）均 **5/15**。另在 32 视频 GT 线索正对照中，真实图 **25/32**、等预算灰图 **12/32**、仅问题 **13/32**。只说明离散地址能改善某种参数读出、已知正确区间内的视觉内容可用；仍未证明自主通信优于文字或改善最终答题。旧 16 视频确认集与这 32 视频都已曝光。 |
| 51 视频单槽真实执行 | 固定问题和强文字基线，只有通过 real / zero / scramble / 同视频 foil / 错问题等控制的地址才允许改窗口。51 个视频中，37 个完整测量、11 个因历史重放数值不一致回退、3 个因已知旧 Source 重放问题回退。24 个被选地址最终产生 **23 个实际改窗、24 个共用合法调用、4 个共用无效**。对 24 个筛选出来的原画面做独立 AI 盲审，真实选中窗相对干预 foil 为 **9 赢、0 输、9 平、6 未决**；9 条获选事实随后判定不在两份原文字报告里。这里的对照是 foil，不能当作赢强文字。 |
| 真实 Focus 及失败定位 | 两臂在 51 个开发视频完成 **70 次**真实 Focus。剔除 Source 已见帧，单槽整窗替换 **16/51**、强文字 **15/51**（4 赢、3 输）；含已见帧时均 **18/51**。官方线索仅作定位指标。加宽窗口没有改善当前所选地址；预算不变的一个互补 shortlist 仅使潜在线索可达从 **7/24→8/24**。两次新进程重放彼此一致，仍复现不了旧档 11 条，不能归因于换卡或用宽松阈值重新放行；同输入的 47 个强文字重跑也有 5 份报告与旧档不同，故主比较保留同进程强文字重跑与相同输入复用。 |
| 51 视频原方案关账 | 实际 Focus 的 **47 视频 / 70 视图**由 AI 视觉盲审，另一批无历史、不同作者的 AI 独立复核 **13 组、125 条事实**。原 51 题的 **65 次**原生 QA 生成已封存并在 CPU 评分，−1.96 个百分点，保守精确 95% 配对区间 **[−11.87, +11.57]** 个百分点。70 个真实视图中 50 个有用，16 个有用视图收到原样的 “No relevant content found.”；后续区分报告弃答与原 query 不匹配。没有独立人类标注一致性估计。 |
| 16+16 改进和三组实际取帧 | 动机：避免 latent 给了新线索、却覆盖掉原文字窗口里已有的线索。三组为：强文字全窗 32 帧、文字锚点 16 + latent 窗 16、相同锚点 16 + 时钟窗 16。后两组共用启用条件；时钟对照隔离第二窗口的选址贡献，不是一个独立部署策略。已曝光 51 题的冻结几何回放：latent 加窗 **18/51**，全量文字和时钟加窗各 **15/51**；逐题对两者均 **4 赢、1 输**。2026-11-22 在 A40 GPU 1 跑完 **51/51 视频、92/92 次 Focus**，审计零失败，时钟组 1 次生成触顶。47 个可执行视频的帧数和视觉 token 逐题完全相同。51 个视频的问答已经封存生成，仍待独立复核后评分。 |
| 单独实现同帧报告改动 | 固定原 query、实际图像顺序/像素和时钟；原样保留输入视觉证据，只允许写出直接可见、带真实帧编号的部分事实，确无证据时输出 `NO_EVIDENCE`。A40 CPU 已用原生预处理核验 **70 对 / 140 份请求**，每臂 **2,211 个图像位置**，两种报告提示的视觉张量与视觉 token 数完全一致。新提示多 44–189 个文字 token。未生成新报告、未做 QA；只能说明干预可执行，不能换掉 16+16 已封存的原生报告。 |

## 恢复工作时先做什么

1. **续收 16+16 候选的视觉盲审。** CPU 导出、全部 19 包下载和验收已完成，无需重跑 92 次 Focus 或原方案 65 次 QA。恢复时先查已有审查者与新写出的回执，再派未审包，避免重复分派。所有 19 个包通过校验后，才把盲审事实和候选原生报告交给另一批独立审查者。候选验收命令和签名在[续跑检查点](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/resume_checkpoint.md)中可查。
2. **完成两阶段盲审，再评分。** 先做视觉，再把事实和匿名报告交给另一批独立、无历史的 AI 逐条核验。16+16 原生报告保留；后续按冻结协议做原生 text-only QA。报告 latent 加窗分别对全量文字与时钟加窗的配对赢、输、净胜和区间。固定 **51 全体、47 合法、23 触发**，四个共用无效计入全体，保留 exact-input 复用；不按观察到的事实结果挑 QA 样本。 三组封存 QA 已完成 51 个视频、71 次实际生成。独立复核已关闭，正式评分见开发集关账。 原评分保持不变；[取证到报告的补充统计](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/blind_triad/flow_diagnostic.md)待事实关账后才运行，分别定位「未采到新事实」与「采到了但未达到报告递送条件」，不据此筛题。
3. **分开验证同帧报告干预。** 对完全相同的原帧和 query，用同一运行会话配对比较旧报告与改后报告，检查新增事实、无依据事实及不变的下游 QA。A40 CPU 已把原视觉盲审的 70 个实际视图、125 条提名全部绑定回配对计划；匿名分包和原文引用核验已用合成样例验收，正式文字包仍为 0。详见[同帧报告复核台账](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/task_plan.md)。两条路线分别记账。
4. **续接首次 Source 捕获与现场取证。** 同会话开关对照已完成并释放单卡，不用重跑此探针。单例已经让文字 Planner 消费本次运行的两份完整报告，并从首次 shared 前向读出地址。基于现场参数重算并真实执行三组窗口的接线也已通过。四视频接线扩展已按[冻结协议](live_source_preflight/live_dev_four_protocol.md)启动，去掉历史状态指纹的放行门槛，并复用同输入的真实分数。[四例均完成并通过独立审计](live_source_preflight/live_dev_four.final.receipt.json)：12 次实际取帧、各 32 帧与 8,064 视觉 token；时钟组 1 次触顶，接收器零视觉编码、无额外 Source 重放。每例 56 次前向、复用省 8 次，选址 22.6–27.1 秒，阶段相加比文字取帧多 5.8–26.3 秒。GPU 已释放，未启动 QA。作为新诊断单独记账；不能替代冻结 16+16 的取证、QA 或成本对照。
5. **保留 96 个未曝光确认视频。** 当前没有动用。旧目录名含 `fresh96` 的数据是已曝光历史。即使开发结果为正，也要先冻结合适的独立协议，才可考虑使用这 96 个视频。

## 证据入口

- [历史参数寻址五种子开发对照](../kev_source_binding/parameter_addressed_comm_20261023/analysis_active_multiseed_r03_natural/analysis-report.md)；[原实验对话：EXP 2 latent 迁移 62.77 rebuttal](codex://threads/01a1714c-1397-7e13-81dc-4a1dc9d05628)。
- [PyTorch 环境镜像交付记录](../qwen35_environment_image_20261109/README.md)；[初期环境迁移记录](../qwen35_video_migration_20261110/migration_status.md)；[A40/A100 固定 101 题完整复核](../qwen35_video_analysis_20261110/fixed101_action_prefix/result.md)；[机制分析与 PCA 图表目录](../qwen35_video_analysis_20261110/analysis-output/figure-catalog.md)（分析中的 A100 分数是运行中快照，最终分数见完整复核）。
- [Introduction 与失败路线图](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/next_action_evidence_diagnosis_20261116/introduction_and_route_map.md)；[8 视频与原目标证据审计](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/full_goal_evidence_audit.md)。
- [51 视频单槽执行报告](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/routing_result.md)；[独立重放诊断](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/replay_stability_v1/result.md)；[原方案盲审规则](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/delivery_review/adjudication.md)；[原方案先生成后开封的 QA 协议](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/qa_sealed/outcome_protocol.md)。
- [16+16 冻结线索对照](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/result.md)；[单卡状态与最终审计](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/sealed_run_status.md)；[三组真实资源审计 JSON](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/resource_audit.summary.json)；[新方案的配对评分规则](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/split_budget_v1/paired_endpoints.md)。
- [旧离散参数机制确认](receiver_training/hard_window_confirmation_a40_20261114/report.md)；[旧实际工具 QA](receiver_training/focus_qa_confirmation_a40_20261114/report.md)；[32 视频 GT 线索正对照](receiver_training/vision_content32_a40_20261115/report.md)。
- [报告改动的独立协议](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/protocol.md)；[70 条实际请求的 CPU 验收](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/original_messages.receipt.json)；[140 份配对输入的实际预处理回执](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/paired_preflight.receipt.json)；[报告对照判读规则](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/endpoints.md)。
- A40 新方案封存目录：`/data1/llj/lljmas/qwen35_cgbench_split_sealed_20261122_02`（`summary.json` 是可阅的结构审计）；原方案取证目录 `/data1/llj/lljmas/qwen35_cgbench_single_slot_focus_20261121_02`，原 QA 封存目录 `/data1/llj/lljmas/qwen35_cgbench_single_slot_qa_sealed_20261122_01`。不在记忆里保留凭据、盲审逐题结果或基准答案。

同帧报告对照已完成 140 次生成、单卡释放和独立递送复核。在状态窗和强文字窗，原提示和“允许部分证据＋引帧”提示均为 0/51，配对净增上下界均为 0。22 个确有新增事实的视图包含 41 条严格机会，两种提示均没写入；修改后总体弃答从 36/70 降到 27/70。递送条件已失败，未启动下游 QA。140 份报告完成陈述抽取，41 个像素盲包的首批已验 3 包、143 条，整体无依据陈述率未关账。见 [同帧报告结果](receiver_training/reusable_strong_text_a40_20261115/postrun_review/decision_value_review/typed_route_parameter_v1/next_hypothesis_review_20261116/source_reliability_followup_20261116/mechanism_revision_20261116/original_protocol_fidelity_20261117_v1/cgbench_source_admission_20261120_v1/native_focus_ceiling_v1/panel_route_v1/failure_analysis_v1/single_slot_v1/report_followthrough_v1/result.md)。

新一轮 `query` 参数对照在单张 A40 上已全部审计并释放。只将历史 Planner 的 `query` 换为公开题干，保留原生报告模板和原图。状态窗确认递送由 0/51 到 1/51，强文字窗从 0/51 到 0/51，两个策略各有 2 个未决视频。125 条提名、250 条盲审已按原规则评分；两种查询各 33/70 弃答。两策略未同时达到预设递送门槛，尚无下游 QA 或确认集结果。见 [同帧 query 结果](focus_query_sameframe_20261124/result.md)。

补审 4 条可能改变 `query` 配对范围的提名后，状态窗净增 [1,1]，强文字窗 [0,1]。未改主评分。另在原 16+16 三条件做了漏报事实补齐正对照：25 条独立复核的事实补入 15 个视图，86 次纯文字 QA 全部审计通过，单卡已释放。三组从 12/51 到 13/51，只救回同一个共用视图视频，23 个触发选窗视频仍是 4/23，状态在两个对照上均 0 赢、0 输。该结果将后续改进具体化为状态寻址能否补到可区分剩余答案的事实，当前未有可独立确认的 QA 增益。见 [漏报事实正对照](fact_delivery_positive_control_20261124/result.md)。

答题差异驱动的 Planner 已完成 47 次单卡纯文字生成并释放，46 条通过格式时间校验，736 个状态布局通过 CPU 检查。94 条匿名评审表明，新请求更常能从短片观察（45/47 vs 42/47），但区分备选下降（26/47 vs 34/47）；四项同时通过由 24/47 降到 15/47，配对 1 赢、8 输。确定输例为 6 个失去区分力、2 个提问暗示答案。后续改为保持原 query，在接收端用真实视觉状态判断哪里能看到区分备选的信息，实际 Focus/QA 仍待验证。见 [取证问题改写结果](answer_discriminating_focus_20261124/result.md)。

新一轮在单张 A40 上执行原生共享 Tool query 的候选答案条件化选窗，2026-11-24 02:26 启动。两路新报告与现时文字 Planner 共用，只有状态条件用第一次 Source 视觉 prefill 的实际状态在控制下选择第二窗。CPU 完成 51 视频／816 布局的验收；02:58 封存并独立复算 13/51（12 个同预算触发、1 个共同无效动作、36 次实际 Focus、28 次唯一纯文字 QA）。时钟组一次 Focus 达到生成上限。全量释放后执行 CPU 审计和匿名画面／报告交接。准确率和相对两个对照的增益尚未计算。见 [运行台账](answer_conditioned_routing_20261124/task_plan.md)。

03:03 同一张 A40 提交 15/51，独立审计覆盖前 13。评分已补上逐视频聚合的三环节诊断：新窗里有没有 Source 未见的区分性事实、报告有没有写入该事实、最终纯文字 QA 有无配对收益。评分合成验证通过，96 个未曝光视频保持封存。

03:12 提交并独立复算 20/51，16 个实际触发，50 次实际 Focus、37 次唯一纯文字 QA。为同轮增加官方线索区间覆盖核验，匹配三臂实际取帧，独立标出 Source 未取到的帧及共享 16 帧以外的贡献。定位和 QA 脚本用 51 题合成数据验收，未读取真实标签或确认集。

03:20 同一张 A40 跑到 25/51，已独立复审前 24。22 个合法案例中选择器实测中位 35.0 秒，选窗加状态取证相对强文字取证中位 +34.9 秒。累计接收端评分 forward 1,192 次，额外 Source 重放与接收端视觉编码均为 0。这是带内容控制的选窗与取证局部成本。未来核对 Introduction 的开销时保留条件及测量范围。见 [实验台账](answer_conditioned_routing_20261124/task_plan.md)。

03:27 同一张卡已提交 30/51。现有 96 个独立 CG-Bench 长视频仍在 A40，媒体数量和总大小与原回执一致，私有名单及题号分配的封存指纹复验一致。该批与历史已曝光 `fresh96` 分开。未解码视频，正式确认和标签封存保持不变。

03:39 前 34/51 个开发视频已独立复算。额外抽验 6 个视频的 479 个实际采样帧，对比帧号/FPS 与解码起始时刻，最大差 38.62 毫秒。线索端点将保留两种时间口径，合成数据已核验近边界的命中变更。真实标签仍封存。

前 38 个视频通过复算。26 个改窗视频中，状态窗相对文字窗/时钟窗分别有 13/14 个输出了完全相同的 Focus 报告，纯文字 QA 因此看不到输入差异。等盲审后检查剩余文案变化中有多少转化为实际新增事实。

04:08 单卡正常生成结束并释放。51/51 全量封存，45 个合法且同预算，35 个实际触发；115 次实际 Focus 和 75 次唯一纯文字 QA。全量审计及匿名包导出交给既有 CPU 等待器，标签尚未打开。见 [实验台账](answer_conditioned_routing_20261124/task_plan.md)。

04:10 全量 51/51 独立 CPU 审计通过。115 次实际 Focus 中 1 次时钟组触顶，75 次唯一纯文字 QA 全部有效且未截断。35 个改窗视频中，状态窗相对强文字/时钟窗分别有 17/18 个生成完全相同的 Focus 文本。原 CPU 等待进程正在按实际模型输入构建匿名画面包，未打开真实标签。

04:38 CPU 导出 23 个匿名画面包，涵盖 45 个有效视频和 115 个真实取证视图，精确画面全部核验。进入独立画面盲审；标准答案、真实线索与确认集尚未开封。

全量区间复核发现 35/35 个已改窗视频的状态第二窗与时钟第二窗无正长度重叠，中心距离中位 604.25 秒，但 18/35 的 Focus 报告逐字相同。它表明动作在时间上确实不同，仍需盲审定位有用事实及报告递送。

06:00 独立 AI 画面盲审已验收 11/23 包、22 个视频、58 个真实取证视图，另三包在审。51 个开发题的题号、题干、选项和未来评分字段的类型与官方标注对齐。未解码答案与线索值，未打开确认集。见 [运行台账](answer_conditioned_routing_20261124/task_plan.md)。

06:57 画面盲审 23/23 包闭环，45 个合法视频和 115 个真实取证视图完整验收。84 条事实交给另一批匿名报告复核。官方开发集 Source 未看过的线索帧命中为状态／强文字／时钟 **19／15／16（各 /51）**，配对 **4 赢 0 输**、**4 赢 1 输**。去掉共享的 16 帧后，实际触发 35 例的命中为 **7／10／1**。两种帧时间口径同结论，事实递送与答题收益仍待报告与 QA 关账。

07:08，报告盲审与官方开发 QA 均关账。状态／强文字／时钟为新可见事实 **14／13／10**、严格报告传递 **0／0／0**、QA **15／16／17**（均 /51）。两项 QA 配对分别 **0 赢 1 输**、**0 赢 2 输**。真实改窗 35 例的 QA 为 **7／8／9**。84 条提名中的 Focus 文本判定 0 条呈现、77 条缺失、7 条未决。先检查准确事实交给现有 QA 的同题正对照，确认集未使用。 [本轮结果](answer_conditioned_routing_20261124/result.md)

本轮事实补齐正对照在单张 A40 完成并释放。39 条事实补入 27 个真实视图，共 101 次纯文字 QA。状态／强文字／时钟同轮补齐前后 **15→15／16→16／16→16**（各 /51），4 次选项变化均为错到错。原生时钟有 1 个未补齐样本未复现，所有改写样本原生选项均复现。下一项先盲审核验事实能否真正排除备选或唯一确定答案，96 个确认视频仍封存。 [正对照结果](answer_conditioned_routing_20261124/fact_oracle/result.md)

27 个补齐视图的选项充分性盲审关账，52 条匿名文本分发到 8 包，评分只打开 17 个对应开发题答案。唯一指向正确答案 **5→6**，新增例在强文字组且原 QA 已正确，状态组无新增。9 个视图排除了新的错误备选且保留正确，6 个排除了官方正确选项。后续把选窗请求收紧为两份共同 Source 报告尚未解决的、可核验的对象与关系差异；保持共同 Tool query、强文字、匹配时钟与原 QA。 [充分性结果](answer_conditioned_routing_20261124/fact_oracle/sufficiency.result.md)

11-24 08:09（北京时间），单张 A40 的可观察目标筛选关账。45 次纯文字生成通过运行审计，GPU 0 已释放。严格解析 0 条通过，事后只规范明确的选项字母串后有 20 条结构通过，65 条独立 AI 匿名判断中仅 2 个新目标五项均通过。缺失目标回退原 query 后，目标策略 **12/45**、原 query **17/45**，同 51 个开发视频 **0 赢、4 输、37 平、10 未决**。本版停在筛选，未做真实选窗和 QA，96 个确认视频仍封存。 [结果和运行路径](answer_conditioned_routing_20261124/observable_target/result.md)

11-24 08:19，在 A40 GPU 0 启动新一轮备选证据读出。接收端扫描从实时 Source prefill 获得的 32 个状态，使用原题、原 query、两份报告和两种选项编号，选出能相对置零增加某个原备选支持的地址，并检验扰乱、错问与错位。三组真实取证与原纯文字 QA 已接线，以强文字及匹配时钟作对照。前 2/51 已封存，GPU worker 2533921 独占单卡，首题已通过独立 CPU 审计。退出后自动复算并评分开发集，96 个确认视频未动。 [运行台账](option_evidence_routing_20261124/task_plan.md) · [协议](option_evidence_routing_20261124/protocol.md)

08:28 新一轮前 3/51 独立验收，首次真实改窗成功，三组共用一个 query 且实际帧序列互异，均为 32 帧、8,064 个视觉 token。时钟组一条报告生成触顶，5 次唯一 QA 无截断或无效；当前没有开封标签。 [前三题回执](option_evidence_routing_20261124/prefix_03.receipt.json)

08:37 新一轮单卡提交 9/51，前 5 题独立复算完成。无标签阶段诊断可定位无候选、弃权、备选重排不稳和控制拒绝，首次实际改窗的状态报告与两对照都不同。已关账旧批次的窗口几何诊断得到固定窗可达 31/51、相邻采样点窗口可达 44/51，但冻结接收器地址回放仍 19/51，35 个改窗例的命中逐题持平。单纯扩大候选窗未给现有地址选择带来收益，当前进程和确认集保持原状态。 [窗口几何诊断](option_evidence_routing_20261124/historical_menu.result.md)

08:39 新一轮 11/51 独立验收，2 次实际改窗、15 次 QA 全部有效且未截断。尚未打开标签。预先对同一批次的已生成 logits 实现了部分证据诊断：两种备选编号都须对同一原备选产生最大正增量，即使可见证据仍不足以直接答题。前 11 题的提名从 4 个变为 9 个，尚无候选准确率或实际取证收益证据。 [前缀审计与方法关联](option_evidence_routing_20261124/task_plan.md)

08:42 新一轮 13/51 独立验收，含 1 次共同无效文字动作，2 次实际改窗，16 次原生 QA 均未截断。部分证据诊断得到 10 个提名、9 个新增或不同地址；真实候选位置和答题质量尚待全量开发评分，进程继续使用一张 GPU。 [当前验收](option_evidence_routing_20261124/prefix_13.receipt.json)

08:51 新一轮提交 18/51，GPU 0 仍由同一进程独占。对首 3 个改窗视频，CPU 复建并核验 96 张 Source、288 张 Focus 的真实输入张量。全量完成后本地可直接收集服务器评分和独立审计，开发集标签仍未开封。 [像素复核](option_evidence_routing_20261124/prefix_15.pixel.receipt.json)

09:00 新一轮提交 23/51、6 次真实改窗，前 20/51 独立验收，待评分进程仍存活。前 20 题的部分证据诊断产生 16 个提名，其中 2 个在已生成分数上被错位对照拒绝、2 个可复用同地址的完整对照、12 个仍需新地址的错问和扰乱测量。实验未追加 GPU 调用、未开标签。 [运行台账](option_evidence_routing_20261124/task_plan.md)

09:05 新一轮提交 27/51，前 25 题独立验收。7 次实际改窗的 224 张 Source 图和 672 张 Focus 图通过实际输入像素审计。独立评分收集器已做配对账本合计与区间验收，服务器锁定评分 payload 未变。 [运行台账](option_evidence_routing_20261124/task_plan.md)

09:08 前 30/51 独立验收，28 个有效且预算匹配、2 个共同无效、8 个实际改窗、44 次原生取证、38 次有效且未截断的 QA。8 个改窗案例中状态报告相对强文字 3 个重复、相对时钟 2 个重复。确认集未动，最终开发评分待全量封存。 [审计](option_evidence_routing_20261124/prefix_30.receipt.json)

09:14 编号审计发现前 8 个实际改窗例中有 1 个的备选在反序后仍用同一输出字母。后续排列已做 CPU 输入验收，保证前 28 个合法请求的 191 个备选全部换字母；未实施 GPU 干预、未改动当前作业。 [验证](option_evidence_routing_20261124/prefix_30.derangement.receipt.json)

09:19 已提交 36/51，前 35 独立审计。9 个改窗里状态组报告相对强文字变化 5 个、相对时钟变化 7 个，原生 QA 选项只分别变化 1 个、2 个。此诊断没有读取正确答案，全部 51 题封存前不报告正误。 [审计](option_evidence_routing_20261124/prefix_35.qa_changes.receipt.json)

09:44 新一轮完成并释放单张 A40。51 个开发视频中 46 个动作有效、12 个真实改窗、5 个共同无效。状态 QA 13/51，强文字和时钟各 12/51；两个优势来自同一例，均 1 赢 0 输、精确单侧 p=0.50。官方新线索三组逐题一致（各 17/51）。部分证据重算加入候选后，给定现有时窗菜单的几何净增上限仍仅 +1/51。当前先追查唯一答题胜例，确认集未开。 [全量结果](option_evidence_routing_20261124/result.md)

09:55 独立机制追查核对唯一胜例，三臂都碰到官方线索，原反序把接收器提名的正确选项留在同一字母。胜例的三个匿名取证视图与三份报告已进入分开复核。GPU 0 在首跑释放后启动 26 个奇数备选视频的单卡重编号对照，比较语义选择与时间地址是否复现，绝不使用确认集或新增 QA。 [台账](option_evidence_routing_20261124/task_plan.md)

10:01 唯一胜例的两项匿名复核关账。状态窗拍到更清楚的鱼群，但 Source 已能辨认鱼与珊瑚，强文字报告也确认了鱼。没有找到状态组独有且能解释改答的新增事实。重新编号对照在同一张卡继续运行，前 7 例原条件动作全复现，其中 1 个真实选址在全换字母时保持相同。 [胜例复核](option_evidence_routing_20261124/win_trace.result.md)

10:16 备选全换字母干预 26/26 完成并释放 A40。原布局 9 个触发，新布局 8 个，共有的 8 个均同选项同地址，原唯一 QA 胜例也复现。该例的编号稳定性已确认，未确认新事实递送或端到端收益。没有消耗确认集。 [干预结果](option_evidence_routing_20261124/layout_stress.result.md)

10:26 单张 A40 启动报告传递实验。锁定上一轮 51 个视频、70 个真正执行的取证视图和 Tool query，提名一条带帧号的可见事实，并用相同模型从引用帧和同视图错位帧分别核验。只把通过的事实补入 QA，同时重跑原报告控制。前 7 题的 6 个核验对由两位独立审查者读图，2 条自动接受的事实均获图片佐证，错位帧均未确认。已提交 41/51，最终准确率未开封。 [协议](grounded_focus_report_20261124/protocol.md) · [台账](grounded_focus_report_20261124/task_plan.md)

10:39 报告传递实验全量完成并释放单张 A40。70 次提名、74 次视觉核验、14 条通过、73 次唯一 QA。状态 13→14、强文字 12→13、时钟 12→13（均 /51），三臂正误翻转均为两个共用视图获益、一个共用视图受损，12 个实际改窗视频逐题零正确性变化。24 条未通过提名给出了超过两帧的引用。14 条机器放行事实已全部发往画面与文本盲审，确认集仍封存。 [结果](grounded_focus_report_20261124/result.md)

10:48 报告干预的全量盲审关账。14 条机器放行事实，9 条有真实画面支持、4 条未决、1 条与画面冲突。仅 1 条同时是原报告缺少且能区分备选的事实，该题三组共用取证，三组都获益。另一个共用视图胜例和共用视图输例，新增可见事实都缺少区别备选的关系。下一诊断比较同一批已开封开发视频上的真实原帧直接读出与等视觉预算灰图，检验当前选窗能否在去掉报告瓶颈时产生稳定差异。 [证据闭环](grounded_focus_report_20261124/result.md)

10:54:25（北京时间），同帧直接读图诊断已在 A40 GPU 0 启动，单卡 25 GiB PyTorch 上限。对锁定的 51 个开发视频复用 70 个真实 Focus 视图，并加入 46 个等视觉预算灰图对照。继续固定两份 Source 报告和题目，检查现有时间参数选到的画面能否帮助答题。10:58 完整处理 10/51，首两题的 128 个输入帧已独立核对。 [运行台账](grounded_focus_report_20261124/task_plan.md) · [诊断协议](grounded_focus_report_20261124/direct.protocol.md)

11:05 同帧直接读图完成并释放 A40 GPU 0。状态窗、强文字窗、时钟窗和灰图在全体 51 个开发视频分别为 **16、15、16、15**，在 12 个真实改窗视频分别为 **5、4、5、6**。状态和时钟窗逐题正误相同，状态对强文字窗的唯一胜例是后者触顶缺答。直接读图相对旧报告的全量净增集中在共用视图。该诊断未提供状态选窗的新增效用证据，后续优先验证选址能否采到区分备选的新证据。原纯文字 QA 结果保留，96 个未曝光确认视频仍封存。 [关账结果](grounded_focus_report_20261124/direct.result.md)

11:15 后，锁定相邻 Source 时间区间及两个真实视觉槽的语义一致性，在 CPU 对原 51 个开发视频做无训练回放。双侧／强文字／匹配时钟／同触发单侧的新线索为 **17／17／16／19**（各 /51），双侧未过筛。后续独立触发的单侧候选暂为 **20／17／16**，但 30 个候选缺内容控制，对强文字净差仍可能为 −1 至 +4，对时钟为 0 至 +4。所有待测候选已由状态选择产生并冻结，补对照后才决定实际取帧。 [实验台账](adjacent_source_routing_20261124/task_plan.md)

11:31 单侧候选的内容对照关账并释放单张 A40。30 个缺控制候选中 29 个通过、1 个回退，3 个数值漂移例全部重测 32 槽后保留原选址。最终 37 个开发视频触发。新线索几何命中 **20／17／16**（各 /51），状态对强文字 **4 赢 1 输**，对同菜单匹配时钟 **4 赢 0 输**。开发筛选通过，冻结后续的同历史原生 Focus 和纯文字 QA，预期 118 次唯一视觉调用。尚无这版真实答题结果。 [对照评分](adjacent_source_routing_20261124/control.score.aggregate.json) · [真实取证预检](adjacent_source_routing_20261124/focus.preflight.receipt.json)

11:33:42，继上一作业释放后使用 A40 GPU 0 启动单侧选窗实际取证。预计 51 题、37 个改窗、118 次唯一 Focus，QA 使用相同模型纯文字生成。首题已提交，监督 PID 2720760，GPU worker PID 2720831。此次固定先前 Source 历史评估工具参数与答题效果，独立 96 题确认集保持封存。 [运行台账](adjacent_source_routing_20261124/task_plan.md)

11:41 实际取证 9/51，24 次视觉调用、15 次唯一文字 QA。首两题的 128 个图像位置和纯文字输出已独立复核，释放后由 CPU 自动评分。额外重放 26 个奇数选项的历史全换字母控制，22 个原提名中仅 12 个保持同选项同地址，仍有布局敏感性。 [当前开发结果](adjacent_source_routing_20261124/result.md)

11:54 真实取证推进至 29/51。前 22 题已重新审计，40 次纯文字 QA 皆有效，强文字选择全部复现。为处理选项顺序和字母敏感性，另在 CPU 实现并预检逐选项 Y/N 证据读出，真问和错问共 632 份提示逐 token 重编号不变。候选文本和数量在错问组保持完全一致。尚未用真实权重验证这版选址或 QA。 [在跑实验台账](adjacent_source_routing_20261124/task_plan.md) · [新读出预检](semantic_support_readout_20261124/preflight.receipt.json)

12:05 相邻 Source 单侧选窗真实实验关账，51/51 全审、118 次视觉、81 次唯一纯文字 QA，A40 GPU 0 已释放。实际线索命中状态／强文字／时钟 **20／17／16**，纯文字 QA **14／12／14**（各 /51）。状态对时钟 QA 1 赢 1 输。状态对两个参照独赢线索为相同的 4 题，3 题 Source 已取过同一官方区间，1 题一起带来 QA 胜例。37 个改窗题里，状态报告与时钟逐字相同 19 份。当前方向是让接收器寻找能区分候选、且未被 Source 文字交代的事实。逐选项读出的前缀缓存虽预估减少约 84% 输入 token，但 BF16 小差异已在完整 CPU 夹具中改变提名，真实筛选以完整前向为默认。此轮未启用 96 个确认题。 [正式关账](adjacent_source_routing_20261124/result.md) · [缓存诊断](semantic_support_readout_20261124/receiver.cache_diagnostic.receipt.json)

12:18:57 逐选项 Y/N 的真实模型筛选已在单张 A40 启动。沿用冻结的 51 个开发题，全部 1,426 个可执行区间通过原生帧序列绑定。首次启动的历史模块同名冲突已用独立包修复，并复查 51 题历史。首题 252 次完整前向通过独立 CPU 审计，12:24 提交 4/51，3 个候选通过内容对照。解封开发标签只在释放及审计后，96 个确认题未动。 [运行台账](semantic_support_readout_20261124/task_plan.md)

12:28 复核前轮实际画面。价格胜例由状态窗补拍的准确字幕支撑，同一官方线索区间的旧 Source 帧展示的是另一支笔。买菜计数题的单个命中帧错过举起大葱的瞬间。使用旧开发标注把区间内 16 帧重排为 8 粗采样加 8 局部采样，即使事后最优中心能增加至少三帧命中，也仅多 1/51 题，暂不启动通用密采。逐选项筛选同一 GPU worker 已提交 7/51 和 1,728 次完整前向，5 个候选通过控制。 [画面与回放核验](adjacent_source_routing_20261124/result.md)

12:41，逐选项真实筛选 20/51、4,572 前向、16 通过，前 18 题独立复核。兼容性预检已精确复现旧 51 题的 118 个 Focus 请求。旧版固定证据状态换时间地址的回放显示，官方线索实际命中 20 题，均匀换窗期望 16.93，整组偏移范围 16–19。聚焦到 Source 未采样过的分段，实际命中 10，均匀换窗期望 9.50，偏移范围 9–11。同时复算相邻左右区间的证据分数换向，有 16 个计划变化，两个线索端点均逐题持平。结果加强了时间绑定对一般线索的现场证据，同时确认其对独有线索的证据仍弱。 [运行与地址诊断](semantic_support_readout_20261124/task_plan.md)

12:44 按用户要求暂停目标并停止 A40 实验，12:45 核实 GPU 0 已释放。逐选项筛选保留 22/51 个已提交记录、18 个通过内容对照的候选和 5,076 次模型前向，尚未评分。 [停止检查点](semantic_support_readout_20261124/task_plan.md)
