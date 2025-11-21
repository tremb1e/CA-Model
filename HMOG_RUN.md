HMOG VQGAN 训练文档

环境准备
- 激活环境：`conda activate tremb1e`（已包含 CUDA）。
- 项目目录：`/data/code/CA-Model`
- 数据目录：`/data/code/ca/refer/ContinAuth/src/data/processed/Z-score_hmog_data_with_magnitude`，包含用户 100669、151985、171538、180679、186676 的 train/val/test CSV。

核心脚本
- `hmog_vqgan_experiment.py`：加载五个用户数据，按 13 个窗口长度（0.1–2.0 秒，50% 重叠）扫参，记录验证 AUC 最佳窗口并可选复训。
- 主要默认参数（可用命令行覆盖）：
  - `--batch-size 128`
  - `--num-workers 22`、`--cpu-threads 44`（充分使用 22 核 44 线程）
  - `--learning-rate 2.5e-4`、`--latent-dim 256`、`--num-codebook-vectors 512`
  - `--use-amp`：默认开启混合精度；多卡自动 DataParallel（除非 `--no-data-parallel`）。
  - 日志 & Checkpoint：`--log-dir results/experiment_logs`，`--output-dir results`

运行示例
```bash
conda activate tremb1e
cd /data/code/CA-Model
# 全部 5 个用户，13 个窗口，每个窗口 1 epoch 扫参，最佳窗口再训 5 轮
python hmog_vqgan_experiment.py --use-amp --sweep-epochs 1 --final-epochs 5
# 只跑某些用户与窗口
python hmog_vqgan_experiment.py --users 100669 151985 --window-sizes 0.5 1.0 2.0 --sweep-epochs 2 --final-epochs 6
```

日志与结果
- 人类可读日志：`results/experiment_logs/hmog_metrics.txt`（包含 stage/user/window/epoch、AUC、FAR、FRR、EER、F1、阈值、推理延迟，全部小数制）。
- 机器可读：`results/experiment_logs/hmog_metrics.jsonl`。
- 训练日志：`results/experiment_logs/hmog_vqgan.log`（Python logging 输出）。
- 最优窗口摘要：`results/experiment_logs/best_windows.json`。
- 最优权重：`results/checkpoints/vqgan_user_<uid>_ws_<window>.pt`。

调参提示
- FAR/FRR/EER 计算在 `compute_metrics` 中通过 ROC 求阈值；若需降低 FAR，可在该函数对 `eer_threshold` 做平移或改用固定阈值策略。
- 推理延迟 `latency` 为单样本平均耗时（秒），从 `evaluate_model` 里获得。

数据与形状
- 输入窗口形状：`(batch, 1, 12, 50)`，包含 12 个传感器通道，时间轴重采样到 50。
- 量化后 token 网格：`6x6`（36 token），匹配 codebook 大小 512，便于 Transformer 使用。

常见问题
- 单卡运行：自动退回 CPU/GPU 可用设备，但速度会下降。
- 若 DataLoader 报内存不足，可降低 `--batch-size` 或 `--max-negative-per-split`。
