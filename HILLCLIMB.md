# HILLCLIMB: 换 lamq 11.2GB 是否值得

## 目标
- 标量：pi 文档 PPL（越低越好），在速度与内存约束内。
- 目标值：[owner] 暂定「PPL 显著低于 8.9GB，且 decode 损失 ≤ 15%」才换。

## 硬约束
- 内存：80K 上下文下进程峰值 ≤ 15.5 GiB（server `--mem-budget` 默认值）；Metal 上限 17.76 GiB。
- 运行时固定：live venv `~/.local/share/lamq/venv`（lamq `ae0ca83` + `lamq.patch`），split-K 开，
  推理时消融开（`lamq/ablate-dirs.npz`）。唯一变量是模型文件。
- 隐私：语料是公开文档，不读本地推理内容。

## Harness（冻结）
- 语料：pi-coding-agent 1.6.0 的 `docs/*.md` 按 C 排序拼接，`~/.cache/inkling-mac/pi-docs-1.6.0.txt`，
  538,224 字节，sha256 前缀 `a1aa57c26d1cfc6a`。与 README 2.732 用的语料不一定相同，所以 8.9GB 重测作基线。
- PPL：`bench.py ppl --corpus … --ablate lamq/ablate-dirs.npz`（8 × 4096，后半段计分）。确定性，n=1，抽查一次可复现。
- 速度 / 内存：`bench.py speed --corpus … --ablate …`，ctx 1K/8K/16K/32K/64K，ndec 128；
  两个模型 A B A B 交替，每个模型 ≥3 次，报中位数和极差。下载等后台负载结束后才测。
- 结果：`bench/ledger.jsonl`，原始输出 `bench/runs/`。

## 预算与停止
- 一次判定：PPL 与速度各一组即收尾。

## 已否定
-

## Open
- lamq 11.2GB 是否值得换：card 报 KL 0.0115（8.9GB 为 0.0208）。owner 2026-11-23 叫停，理由是新模型更新快。
  最快验证：下载 `522cd63`（11.2 GB，本机约半小时），跑 `bench.py ppl`，和 2.7320 比；再 A/B 跑 `speed`。
  runtime 已支持它用到的 nvfp4 格式（live venv 里有 nvfp4 kernel），未实际加载过。

## 变更记录
- 2026-11-23 建立。
- 2026-11-23 停止（owner 叫停）。只测了 8.9GB 基线 PPL 2.7320。llama.cpp 备用模型 AD-IQ3_S 已删，mmproj 保留（lamq 视觉要用）。
