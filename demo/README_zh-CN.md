# Sage-Interaction Demo

[English](README.md) | [简体中文](README_zh-CN.md) · [项目介绍](../README_zh-CN.md)

## 安装

需要 Linux、支持 BF16 的 NVIDIA GPU、Python 3.12 和 CUDA 12.8 工具链。

从仓库根目录执行：

```bash
cd demo
python3.12 -m venv .venv
source .venv/bin/activate
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp" PIP_NO_CACHE_DIR=1
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchaudio==2.8.0
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/prepare_environment.py
```

## 配置

```bash
cp config.example.json config.local.json
```

修改 `config.local.json` 中的以下字段。模型和音频路径相对于配置文件所在目录。

| 字段 | 设置 |
| --- | --- |
| `checkpoint` | 模型目录，包含权重、tokenizer 和模型代码；默认 `models/sage-interaction` |
| `tts_base` | 包含 `assets/token2wav/` 和基础 TTS 权重的目录，默认使用 checkpoint |
| `prompt_wav` | 音色参考 WAV，默认使用 checkpoint 中的 `system_ref_audio.wav` |
| `backend_base_url` | 后台 API 地址，需支持 `/models`、`/chat/completions` 和结构化 JSON 输出 |
| `backend_model_name` | 后台模型名称 |

## 启动

```bash
bash start_demo.sh --config config.local.json --gpu 0 --instances 1
```

按隐藏提示输入后台 API key，或在启动前设置环境变量 `SAGE_BACKEND_API_KEY`。

出现 `Demo ready` 后，打开 [http://127.0.0.1:18080/](http://127.0.0.1:18080/)，允许麦克风权限。按终端中的 **Ctrl+C** 停止。

两个 GPU 各运行一个实例时，使用 `--gpu 0,1 --instances 2`；用 `--port` 修改访问端口。查看全部参数：

```bash
bash start_demo.sh --help
```

## 远程访问

在自己的电脑上执行，将 `demo-host` 替换为 GPU 主机的 SSH 别名：

```bash
ssh -NT -L 18080:127.0.0.1:18080 demo-host
```

随后在浏览器打开 [http://127.0.0.1:18080/](http://127.0.0.1:18080/)。
