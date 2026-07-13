# Qwen3-ASR-MLX-Server

[English](./README.en.md) | 中文

一个面向 macOS 的本地 FastAPI 服务，基于 `mlx-audio` 与 Qwen3-ASR（MLX），提供 Whisper / OpenAI 兼容的本地转录接口。

兼容以 MacWhisper 为代表的转录应用，补齐 `verbose_json`、`segments`、`words`、词级时间戳和语言字段等常用响应结构，并处理 MP3 / MP4 / M4A 等输入转码。

服务可直接用于 Spokenly、OpenWhispr 以及其它 OpenAI 兼容转录客户端。在本地实测中，普通话听写速度非常快，质量接近阿里云官方转录服务。

## 功能特性

- Whisper 兼容转录端点
  - `POST /v1/audio/transcriptions`
  - `POST /audio/transcriptions`（别名）
- Chat 兼容音频转录端点
  - `POST /v1/chat/completions`
  - `POST /chat/completions`（别名）
- 健康检查与模型列表端点
  - `GET /healthz`
  - `GET /v1/models`
- 支持 multipart 音频和 JSON `audio_file`（base64/buffer/list 变体）
- 可选 forced alignment（词级时间戳，基于 Qwen3-ForcedAligner）
- 支持终端交互式 setup 流程

## 模型策略

本项目仅支持 Qwen3-ASR 的 MLX 命名（全精度与量化版本），例如：

- `qwen3-asr-mlx`
- `qwen3-asr-1.7b-bf16`
- `qwen3-asr-1.7b-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

若模型文件缺失，API 会返回明确的 setup 提示。服务不会在请求时自动下载模型。

## 使用指南

### 1）克隆仓库

```bash
git clone https://github.com/Jason-Queen/Qwen3-ASR-MLX-Server.git
cd Qwen3-ASR-MLX-Server
```

### 2）安装

#### 默认：uv

```bash
uv sync --python 3.11
```

如果你还没有 Python 3.11，可先执行：

```bash
uv python install 3.11
uv sync --python 3.11
```

#### 兼容方式：Conda

```bash
conda env create -f environment.yml
conda activate qwen3-asr-whisper
```

#### 兼容方式：venv / pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
```

### 3）准备模型（转录前必做）

交互式 setup（可选择 Hugging Face 或 ModelScope）：

```bash
uv run --python 3.11 qwen3_asr_mlx_server.py setup
```

非交互示例：

```bash
uv run --python 3.11 qwen3_asr_mlx_server.py setup --non-interactive --source huggingface
```

自定义你想下载的模型（交互模式）：

- 在 setup 过程中，按提示输入你自己的 `Repository id`（可填 bf16 或量化仓库）。
- 同时可自定义本地目录（`Local directory`）。
- 服务最终加载的是你指定目录里的模型文件。

常见仓库示例（Hugging Face）：

- `mlx-community/Qwen3-ASR-1.7B-bf16`
- `mlx-community/Qwen3-ASR-1.7B-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

主动下载 Aligner（词级时间戳）：

- 交互模式：运行 `uv run --python 3.11 qwen3_asr_mlx_server.py setup`，在 `Qwen3-ForcedAligner-0.6B-bf16` 提示处选择 `y`。
- 你也可以单独再跑一次 setup，只下载缺失的 Aligner。
- 可通过 `--source huggingface` 或 `--source modelscope` 指定下载源。

### 4）启动服务

```bash
uv run --python 3.11 qwen3-asr-mlx-server
```

默认监听：`127.0.0.1:8989`

局域网模式（显式开启）：

```bash
QWEN_MLX_HOST=0.0.0.0 QWEN_MLX_PORT=8989 uv run --python 3.11 qwen3-asr-mlx-server
```

安全提示：

- 默认只监听本机，避免误暴露服务。
- 只有在需要局域网接入时才开启 `0.0.0.0`。
- 如果机器有公网 IP 或端口映射，请不要直接暴露该服务到公网。

指定加载你自己的模型目录：

```bash
QWEN_MLX_MODEL_PATH=/path/to/your/qwen3-asr-model \
QWEN_MLX_ALIGNER_PATH=/path/to/your/qwen3-aligner-model \
uv run --python 3.11 qwen3-asr-mlx-server
```

如果你使用 Conda 或 `venv`，现有启动命令仍然兼容：

```bash
python qwen3_asr_mlx_server.py
```

说明：

- API 请求里的 `model` 字段用于兼容与校验；
- 实际加载的模型由 `QWEN_MLX_MODEL_PATH` 决定。
- 默认服务不会在启动或请求过程中自动下载模型；若自动识别不到缓存目录或本地目录，请显式设置
  `QWEN_MLX_MODEL_PATH` / `QWEN_MLX_ALIGNER_PATH`，或手动运行 `python qwen3_asr_mlx_server.py setup`。

### 5）快速测试

```bash
curl -X POST 'http://127.0.0.1:8989/v1/audio/transcriptions' \
  -F "file=@/path/to/test.wav" \
  -F "model=qwen3-asr-1.7b-bf16" \
  -F "response_format=json"
```

### 6）关键环境变量

```bash
QWEN_MLX_MODEL_PATH=./Qwen3-ASR-1.7B-bf16
QWEN_MLX_ALIGNER_PATH=./Qwen3-ForcedAligner-0.6B-bf16
QWEN_MLX_MODEL_ID=qwen3-asr-mlx
QWEN_MLX_HOST=127.0.0.1
QWEN_MLX_PORT=8989
QWEN_MLX_MAX_UPLOAD_MB=100
QWEN_MLX_MAX_QUEUE_DEPTH=8
QWEN_MLX_ALIGNMENT_CHUNK_SECONDS=30
QWEN_MLX_ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS=5
QWEN_MLX_AUTO_ALIGN_LANG_CODES=zh,en,ja,ko
QWEN_MLX_LOG_PROMPTS=0
```

其中 `QWEN_MLX_LOG_PROMPTS=1` 会显式开启请求 prompt 预览日志，默认关闭；也可以用启动参数：

```bash
uv run --python 3.11 qwen3-asr-mlx-server --log-prompts
```

分片时长调优建议：

- `30`（默认，推荐）：作为目标窗口，时间戳精度和识别稳定性更平衡；实际切点默认会在 5 秒范围内优先吸附到 ASR segment 结尾，避免切断句子。
- `15`（更激进）：时间戳可能更细，但速度更慢，且上下文更碎。
- `60+`：速度更快，但时间戳通常更粗。
- `QWEN_MLX_ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS=0` 可以恢复固定秒数硬切。

## 说明

- 仅支持 `task=transcribe`；`translate` 会被拒绝。
- 为避免 MLX 原生层在并发下崩溃，单进程内的转录与 forced alignment 推理会串行执行。每个进程允许 1 个执行中的请求和 `QWEN_MLX_MAX_QUEUE_DEPTH` 个 in-flight 等待请求；超出容量时返回 `429` 与 `Retry-After: 1`。
- `/healthz` 返回 ASR 模型、可选 aligner、admission/队列 readiness，以及 admission、推理锁等待、解码音频内存、进程 peak RSS 和取消累计指标。必需的 ASR 模型未就绪或 admission 已满时返回 `503`；可选 aligner 不可用会明确显示，但不会阻止纯文本转录。
- 普通 `json` 与 `text` 是纯文本契约。对这两种格式显式设置 `alignment=on`、`word_timestamps=true` 或 `timestamp_granularities` 时返回 `400`，不会再执行对齐后隐藏结果。JSON 时间戳请使用 `verbose_json`，字幕请使用 `srt`/`vtt`；Chat Completions 仍为纯文本响应并拒绝时间戳输出控制。
- 自动 forced alignment 只会用于可消费时间戳的输出（`verbose_json`、`srt`、`vtt`）。
- `zh/en/ja/ko` 的自动 forced alignment 会按运行时依赖自动判定：
  - `zh/en`：需要 aligner 模型可用
  - `ja`：需要 aligner + `nagisa`
  - `ko`：需要 aligner + `soynlp`
- `verbose_json` 响应里：
  - `language` 表示主语种，不表示全文件唯一语种
  - `primary_language` 是与 `language` 等值的显式别名
  - `segments[].language` 表示局部分段语种
  - `detected_languages[]` 表示基于分段聚合的语种及时长摘要
  - `timestamp_provenance` 标明 segment 时间来自 ASR、估算还是 forced alignment，同时给出词时间来源与 alignment 状态
- 请求若在等待进程内推理锁时取消，会退出等待队列；已经开始的原生 MLX 调用会先完成再删除临时音频，请求仍以取消结束，并按阶段计入指标。

## 警告

在生产环境使用前，请仔细阅读源代码，并先在你自己的环境中完成验证。

## 鸣谢

感谢阿里巴巴、苹果 MLX 项目、MLX-Community 以及更广泛的开源生态。

## 版权与协议

- Copyright (c) 2026 Qwen3-ASR-MLX-Server contributors
- 本仓库采用 Apache-2.0 协议（见 `LICENSE`）
- 第三方依赖保持其各自原始协议
- 通过 Hugging Face / ModelScope 下载的模型权重与 tokenizer，仍受其上游协议与使用条款约束
