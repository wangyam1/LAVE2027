# 96 视频独立盲确认台账（`confirm96_20261126`）

- [x] 核验 A40 封存 96 长视频确认资产（`roster.private.json`, `qid_roles.private.json`, `integrity.private.json`, `cgbench.json` SHA-256 全部吻合且与 51 开发视频 0 重叠）
- [x] 编写并本地/远程 CPU 单测冻结协议执行器（`confirm_inputs.py`, `confirm_run.py`, `confirm_audit.py`, `confirm_score.py`, `finalize.sh`, `test_confirm_cpu.py` — 4/4 通过）
- [x] 运行无 GPU 盲预检（`preflight` PIN `45c81f11669e51462a3d80c2d1a3507a7bbccf6eb62fb0092160b57c6201b6c5`，96/96 MP4 完整校验、`labels=0, clues=0`）
- [x] 单卡独占 A40 GPU 0 执行 96/96 全量推理并通过前 2 例独立 CPU 前缀审计（峰值显存 19.95 GiB < 25 GiB 上限）
- [x] 校验 GPU 0 彻底释放至 `0 MiB`，独立 CPU 审计生成 `release.json`（`f8366fd6...`），一次性解封生成 `score.json`（`4fc650f0...`）
- [x] 同步本地回执、完成解封后只读根因诊断并归档至 `notes.md` 与 `RESEARCH_MEMORY.md`
