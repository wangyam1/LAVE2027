# LongVideoBench 150 类别均衡全量评测记录（`lvb_vllm_20261126`）

- **完成时间**：2026-11-26 08:39 CST（Shard 1 于 08:26 结束、Shard 0 于 08:39 结束）
- **远端目录**：`/data1/llj/lljmas/qwen35_lvb150_vllm_20261126_01`
- **硬件与环境**：A40 GPU 0 + GPU 1 双卡分片（每卡 75 题，`batch_cases=4`，`gpu_memory_utilization=0.90`），隔离环境 `/data1/llj/lljmas/envs/lave_vllm_20261126`（只读克隆 `toolrl2`：`vllm 0.19.0`, `torch 2.10.0+cu128`, `transformers 5.3.0`, `flash_attn 2.8.3`, `fla 0.4.2`），冻结 `Qwen3.5-9B`，100% Training-Free（`temperature=0.0`, `thinking=False`）。**两卡均已完全释放至 `0 MiB`（`gpu_released: true`）。**
- **不可变回执与校验和**：
  - `preflight.json`: `27b9d8338d198ec2bcd18d846ce7eaf3bae024d244a93692507c67b194d8ec08`（150 样本，146 个独立长视频，`duration >= 180s`，覆盖全部 17 个 LVB 题型且每类 8–9 题；`duration_group=3600` 101 题、`600` 49 题）
  - `shard_0.terminal.json`: `6832c1f901480937b0865cbea47de4bd78afecbc7992377f3316e86bd86471e8`（GPU 0，75 题；第 44 题遇到 vLLM 0.19.0 默认 4 GiB `mm_processor_cache` 驱逐不同步后，补丁 `mm_processor_cache_gb=0` 断点续跑完成余下 31 题）
  - `shard_1.terminal.json`: `21cfed27464711ee9756b8538b36093e9b85922fc16221df28eca0701432259f`（GPU 1，75 题，全程零报错）
  - `score.json`: `e5c5cfe129a576385c32c1e439db36493944ae88baa76354b2be8d7776282d54`（CPU 单次解封评分）

---

## 1. 控制通过率、轮次帧数与运行耗时

- **控制通过率**：`137 / 150` 合法文字 Planner 动作（13 例非法动作三臂统一回退），其中 `130` 例进入 Receiver 扫描，**`108 / 130`（`83.1%`）通过全部三项严格内容控制**（与 CG-Bench 96 盲确认的 `69 / 83 = 83.1%` 完全一致）。
  - 区间数分布：`K=4`: `101`、`K=2`: `5`、`K=1`: `2`、`K=0`: `42`（控制拒绝 `20`、无提名 `2`、未扫描 `20`）。
- **平均视觉轮次与帧数（`N = 150`）**：
  - Focus-1：**`1.91` 轮 / `60.71` 帧**（Source 32 帧 + 合法题 Focus-1 32 帧）。
  - Focus-1+2（`equal_budget`）：**`2.37` 轮 / `67.96` 帧**（`68 / 150` 题触发 Focus-2，新增 16 张密集帧）。
- **vLLM-LAVE 均题阶段耗时（三臂同跑，4 题合批摊销）**：
  - 全流程总计 **`70.28 s`**（较 HF 版 CG-Bench `138.61 s` 提速约 **2.0×**，且额外包含了 6 路 `direct_visual_qa` 前向）；
  - **LAVE Receiver（33 槽 + 3 项对照，零视觉编码）**：仅 **`5.07 s`**；
  - `source_shared`: `21.71 s`（Source 合计 `33.12 s`）、`planner`: `1.28 s`、`focus1`（三臂）: `21.08 s`、`focus2`（三臂）: `6.71 s`、`qa1`: `0.38 s`、`qa2`: `0.09 s`。

---

## 2. 全量 150 题端到端准确率汇总

| 评测口径（`N = 150`） | `text` 基线 | **`state_split` (LAVE)** | `clock_split` | `state vs text` 配对（双侧精确 McNemar `p`） | `state vs clock` 配对 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`direct_visual_qa_f2` (`op`/`eq`同)** | `83` (`55.3%`) | **`87` (`58.0%`)** | `85` (`56.7%`) | **`+5 / −1`（净 `+4`, `+2.67 pp`, `p = 0.219`）** | `+4 / −2`（净 `+2`, `+1.33 pp`, `p = 0.688`） |
| `direct_visual_qa_f1` | `84` (`56.0%`) | **`85` (`56.7%`)** | `84` (`56.0%`) | `+3 / −2`（净 `+1`, `+0.67 pp`, `p = 1.000`） | `+3 / −2`（净 `+1`, `+0.67 pp`, `p = 1.000`） |
| 纯文本 `focus1_native` | `80` (`53.3%`) | `81` (`54.0%`) | `82` (`54.7%`) | `+1 / −0`（净 `+1`, `+0.67 pp`, `p = 1.000`） | `+0 / −1`（`p = 1.000`） |
| 纯文本 `focus12` (`op`/`eq`同) | `80` (`53.3%`) | `81` (`54.0%`) | `81` (`54.0%`) | `+1 / −0`（净 `+1`, `+0.67 pp`, `p = 1.000`） | `+1 / −1`（`p = 1.000`） |

### 关键分层结果（`direct_visual_qa_f2`）
- **108 题 Focus-1 实际改道子集（`state != text`）**：`state_split` **`71 / 108` (`65.7%`)** vs `clock_split` `69 / 108` (`63.9%`) vs `text` `67 / 108` (`62.0%`)——LAVE 相对 `text` **`+3.70 pp` (`+5/−1`)**，相对 `clock_split` **`+1.85 pp` (`+4/−2`)**。
- **137 题合法动作子集**：`state_split` **`87 / 137` (`63.5%`)** vs `clock_split` `85 / 137` (`62.0%`) vs `text` `83 / 137` (`60.6%`)（`+2.92 pp`）。
- **按视频时长分组**：
  - **10–60 分钟超长视频（`duration_group = 3600`, `n = 101`）**：`state_split` **`57 / 101` (`56.4%`)** = `clock_split` `57 / 101` (`56.4%`) > `text` `54 / 101` (`53.5%`)，`state vs text` **`+4 / −1`（`+2.97 pp`, `p = 0.375`）**；
  - **3–10 分钟长视频（`duration_group = 600`, `n = 49`）**：`state_split` **`30 / 49` (`61.2%`)** > `text` `29 / 49` (`59.2%`, `+1/−0`) > `clock_split` `28 / 49` (`57.1%`, **`+2/−0`**)；且单步 `direct_visual_qa_f1` 上 `state_split` 已达 **`30 / 49` (`61.2%`)**，同时领先 `text` 与 `clock` 各 `+2/−0`（`+4.1 pp`）。

### 17 题型覆盖情况（`direct_visual_qa_f2_op`，`state_split` vs `text`）
- **17 个题型零退化**：4 个题型严格领先（`E2O`: 4 vs 3、`O2E`: 7 vs 6、`S2O`: 5 vs 4、`T2E`: 4 vs 3），13 个题型持平，**0 个题型落后**。
- **逐题翻盘清单（`state_split` vs `text`）**：
  - 5 例净胜：`ord=19` (`duxO1EZ650E_0`, `O2E`, 3600s, `C vs A`)、`ord=23` (`DoizYSYQRqU_0`, `S2O`, 3600s, `A vs B`)、`ord=34` (`fZBC3nmvJb8_0`, `E2O`, 3600s, `C vs D`)、`ord=50` (`XR3Ov2nQ39s_0`, `TOS`, 600s, `D vs C`)、`ord=130` (`MJYBHfYF8LI_1`, `T2E`, 3600s, `C vs D`)；
  - 1 例净负：`ord=135` (`wvfctNd-Aio_1`, `TOS`, 3600s, `E vs A`——该题在纯文本 `qa1` 中反而由 `state` 答对 `A vs E`)。

---

## 3. 核心机制发现与下一步

1. **双通道瓶颈完全定量拆解**：
   - 仅改路由但保留 Focus→纯文本摘要（`text` 纯文本 `80/150 = 53.3%` → `state` 纯文本 `81/150 = 54.0%`）：在 108 个改道视频中，纯文本 `qa1` 仅在 **`4 / 108`（3.7%）** 题上改变了选项字母（`qa2`: `5 / 108`），证明 Focus→纯文字摘要几乎吃掉了全部视觉改道差异。
   - 移除 Focus→纯文本压缩（改为 `direct_visual_qa_f2`）：基线 `text` 从 `53.3%` 升至 `55.3%`（`+2.0 pp`），而 LAVE `state_split` 从 `54.0%` 升至 **`58.0%`**（`+4.0 pp`），叠加 LAVE 状态路由贡献 **`+2.67 pp`**（`+5/−1`），合计较原纯文本基线净增 **`+4.67 pp`（`80 → 87 / 150`）**。
2. **诚实统计边界**：在 2 步短链（均 `2.37` 轮）下，`direct_visual_qa_f2` 的 `+5/−1` 双侧精确 `p = 0.219`（单侧 `0.109`），方向一致且 17 类零退化，幅度已超过 7B 有训练蒸馏工作 `LARVA (2610.35229)` 在 LongVideoBench 上的 `+1.3 pp`，但因 108 次改道中仅 6 题改变最终选项，单次路由的方差尚未穿破 `p < 0.05`。
3. **下一步**：接入官方多步 **VideoSeek** 闭环（`overview / skim / focus / answer`，`max_steps = 6–8`，纯单视频自检无跨视频 foil），在同底座 `Qwen3.5-9B` + vLLM 上正面对比 `Vanilla VideoSeek` vs `VideoSeek + LAVE`（以及 `Video-Holmes` 与 `Qwen3-VL-Embedding-2B` 检索基线）。
