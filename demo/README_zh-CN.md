# Sage-Interaction Demo

[English](README.md) | [简体中文](README_zh-CN.md) · [项目介绍](../README_zh-CN.md)

在浏览器中使用 Sage-Interaction 进行实时语音对话、异步任务委托和 Time Speak 定时提醒。每个 worker 独占一个模型实例，同一时间处理一条会话。

默认**关闭 VAD 打断检测，开启定时提醒**，保留模型原生全双工交互和原有麦克风降噪。摄像头按钮提供本机画面预览；当前语音 demo 不向模型传输视频帧。

## 环境与模型

- Linux、支持 BF16 的 NVIDIA GPU，以及足够加载模型的可用显存。建议从一个实例开始。
- Python 3.12；已核对的运行环境使用 PyTorch 2.8.0、CUDA 12.8。安装 FlashAttention 需要兼容的 CUDA 工具链和编译工具。
- 本地 Sage-Interaction checkpoint，包含模型代码、tokenizer、safetensors 索引和权重分片。
- `assets/token2wav/` 下的原生语音资产：`flow.pt`、`flow.yaml`、`hift.pt`、`campplus.onnx` 和 `speech_tokenizer_v2_25hz.onnx`。
- 音色参考 WAV，默认使用 checkpoint 中的 `system_ref_audio.wav`。
- 可用的推理后台，支持 `/models`、`/chat/completions`、结构化 JSON 输出及 `sage_demo/protocol.py` 中的请求选项。

本目录不包含模型权重和语音录音，需要单独准备。

## 安装

从仓库根目录执行：

```bash
cd demo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/prepare_environment.py
```

最后一步为 `stepaudio2-minicpmo==0.1.1` 应用两处已有兼容调整：精简 Token2wav 导入，以及当前 demo 使用的首个语音块补齐逻辑。脚本校验文件哈希，可重复执行；遇到无法识别的修改会停止。补丁位于 `patches/`，只在本 demo 的虚拟环境中执行。

## 配置与启动

```bash
cp config.example.json config.local.json
```

修改 `config.local.json` 中的模型和后台配置。配置里的路径相对于配置文件所在目录，默认模型目录为 `models/sage-interaction`；随项目提供的 prompt 相对于 demo 目录解析。命令行传入的路径相对于当前工作目录。

语音资产单独存放时，设置 `tts_base`，或传入 `--tts-base models/base-model`；checkpoint 不含 TTS 权重时，也从该基础模型加载。音色参考单独存放时，设置 `prompt_wav` 或使用 `--prompt-wav`。

```bash
# 仅检查配置与模型文件，不访问 GPU 和网络。
bash start_demo.sh --config config.local.json --check-config

# 检查依赖、GPU、端口和后台访问，不加载模型权重。
bash start_demo.sh --config config.local.json --check-only

# 在 GPU 0 上启动一个实例。
bash start_demo.sh --config config.local.json --gpu 0 --instances 1
```

按隐藏提示输入后台 API key。无人值守启动时，通过进程环境提供 `SAGE_BACKEND_API_KEY`。在交互式 Bash 中可使用以下方式，避免将密钥值写入命令历史：

```bash
read -rsp 'Backend API key: ' SAGE_BACKEND_API_KEY
echo
export SAGE_BACKEND_API_KEY
```

密钥只供服务器进程使用，不写入运行配置 JSON，也不发送到浏览器。可以用 `SAGE_BACKEND_BASE_URL` 和 `SAGE_BACKEND_MODEL` 覆盖后台设置；不同服务商的专有请求选项需要自行适配。

出现 `Demo ready` 后，打开 **http://127.0.0.1:18080/**，允许麦克风权限，戴上耳机并点击开始语音对话。页面也支持文字输入和数学问题示例。保持启动终端运行，按 **Ctrl+C** 停止本次启动的 worker 和网关。

## 访问远程 GPU 主机

在 GPU 主机上启动 demo。在自己的电脑上执行端口转发，将 `demo-host` 替换为自己的 SSH 主机别名：

```bash
ssh -NT -L 18080:127.0.0.1:18080 demo-host
```

随后在电脑浏览器打开 **http://127.0.0.1:18080/**。服务默认监听回环地址；通过 localhost 访问也满足浏览器申请麦克风、摄像头权限的安全上下文要求。

## 常用参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--config` | 内置示例 | 本机 JSON 配置 |
| `--checkpoint` | `models/sage-interaction` | 模型权重及模型代码 |
| `--tts-base` | checkpoint 目录 | 单独存放的语音资产及基础 TTS 权重 |
| `--prompt-wav` | checkpoint 中的 `system_ref_audio.wav` | 音色参考 |
| `--gpu` | `0` | 逗号分隔的 GPU 编号或 UUID |
| `--instances` | `1` | 独立实例数，轮流分配到所选 GPU |
| `--vad` | `off` | 额外的 VAD 打断检测 |
| `--port` | `18080` | 浏览器访问端口 |
| `--worker-port` | `19310` | worker 起始端口，后续实例依次递增 |
| `--run-dir` | `runs/` 下的新目录 | 本次运行输出，指定目录必须为空 |
| `--ready-timeout` | `1800` | 模型启动超时，单位秒 |

两个 GPU 各运行一个独立实例：

```bash
bash start_demo.sh --config config.local.json --gpu 0,1 --instances 2 --vad off
```

## Time Speak

可以说：**“Remind me to stretch in 60 seconds.”** 等待时继续对话。前端委托明确的相对时间后，后台异步计时，到期返回 `time_up` 事件。

当前解析器识别 `in 60 seconds`、`after 2 minutes` 等英文相对时间表达。默认最多提前 840 秒设置提醒，单次会话最多持续 900 秒。提醒到达前需保持会话；关闭会话会取消待处理任务。当前计时器尚未实现周期提醒、单独修改或取消某个提醒。

## 文件与运行输出

`start_demo.sh` 是启动入口；`launch.py` 负责配置检查和进程管理；`sage_demo/` 包含推理 worker、任务委托、定时提醒、网关及浏览器页面；`scripts/` 与 `patches/` 负责环境准备；`tests/` 提供无需加载模型的回归检查。

每次启动创建独立的 `runs/demo-…/` 目录，保存运行配置、日志和会话音频与事件。会话文件保留在本机。`.gitignore` 排除运行目录、本机配置、环境、模型和凭据。

## 排查与验证

- 模型文件缺失：用 `--check-config` 核对 checkpoint、`tts_base` 和音色参考。
- 环境不匹配：按依赖版本安装，执行 `python scripts/prepare_environment.py --check`。
- GPU 不可用或显存不足：确认在 GPU 主机上运行，检查卡号，先使用一个实例。
- 端口占用：调整浏览器端口和 worker 端口范围。
- 后台检查失败：检查密钥、API 地址、模型名称和主机网络。检查过程不提交推理请求。
- 无麦克风权限：通过 localhost 或正确配置的 HTTPS 访问。
- worker 启动失败：查看 `runs/demo-…/logs/worker-0.log`；启动失败会清理本次创建的进程。

运行不加载模型权重的回归检查：

```bash
python -m unittest discover -s tests -v
```
