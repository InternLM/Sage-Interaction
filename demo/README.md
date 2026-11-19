# Sage-Interaction Demo

[English](README.md) | [简体中文](README_zh-CN.md) · [Project overview](../README.md)

Run a browser-based voice conversation with Sage-Interaction, asynchronous task delegation, and Time Speak reminders. Each worker owns one model and processes one conversation at a time.

The default configuration disables **VAD-based interruption detection** and enables **scheduled reminders**. The model's native full-duplex interaction and the existing microphone noise filter remain enabled. The camera control provides a local preview; this voice demo does not send video frames to the model.

## Requirements

- Linux and an NVIDIA GPU with BF16 support and enough available memory for the selected checkpoint. Start with one worker.
- Python 3.12. The validated environment uses PyTorch 2.8.0 with CUDA 12.8; installing FlashAttention also requires a compatible CUDA toolkit and build tools.
- A local Sage-Interaction checkpoint with its model code, tokenizer, safetensors index and weight shards.
- Native speech assets under `assets/token2wav/`: `flow.pt`, `flow.yaml`, `hift.pt`, `campplus.onnx`, and `speech_tokenizer_v2_25hz.onnx`.
- A voice-reference WAV file. By default, the launcher looks for `system_ref_audio.wav` in the checkpoint directory.
- Access to the configured reasoning backend. It must support `/models`, `/chat/completions`, structured JSON output, and the request options in `sage_demo/protocol.py`.

Model weights and voice recordings are not distributed in this directory. Obtain the checkpoint and speech assets separately before starting the demo.

## Install

From the repository root:

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

The last command applies two checked adjustments to `stepaudio2-minicpmo==0.1.1`: a minimal Token2wav import and the first-chunk speech padding used by this demo. It verifies file hashes, accepts an already prepared environment, and rejects unrecognized local changes. The patches are in `patches/`; run this command inside the demo's virtual environment.

## Configure and start

```bash
cp config.example.json config.local.json
```

Edit `config.local.json` for your model and backend. Paths in this file are relative to the configuration file's directory. The default checkpoint location is `models/sage-interaction`. Prompt files bundled with the demo are resolved relative to the demo directory. Paths supplied on the command line are relative to your current working directory.

If speech assets are stored in a separate base-model directory, set `tts_base` or pass `--tts-base models/base-model`. Checkpoints without TTS weights also load those weights from that base model. Set `prompt_wav` or use `--prompt-wav` when the voice reference is stored separately.

```bash
# Check configuration and model files without loading a model or making API calls.
bash start_demo.sh --config config.local.json --check-config

# Check dependencies, GPUs, ports, and backend access without loading model weights.
bash start_demo.sh --config config.local.json --check-only

# Start one worker on GPU 0.
bash start_demo.sh --config config.local.json --gpu 0 --instances 1
```

Enter the backend API key at the hidden prompt. For unattended startup, provide `SAGE_BACKEND_API_KEY` through the process environment. In an interactive Bash terminal, it can be set without placing the value in shell history:

```bash
read -rsp 'Backend API key: ' SAGE_BACKEND_API_KEY
echo
export SAGE_BACKEND_API_KEY
```

The key is used only by server processes and is never written to the runtime JSON or sent to the browser. `SAGE_BACKEND_BASE_URL` and `SAGE_BACKEND_MODEL` can override the backend settings. Use your own endpoint and model configuration; provider-specific options are not automatically translated.

When the launcher prints `Demo ready`, open **http://127.0.0.1:18080/**. Allow microphone access, use headphones, and click the microphone button. Text input and a mathematical question example are also available. Keep the launcher terminal open; **Ctrl+C** stops its worker and gateway processes.

## Access a remote GPU host

Start the demo on the GPU host. From your own computer, forward its browser port, substituting your SSH host alias for `demo-host`:

```bash
ssh -NT -L 18080:127.0.0.1:18080 demo-host
```

Open **http://127.0.0.1:18080/** on your computer. The gateway and workers bind to loopback. Localhost access through the tunnel also provides the secure browser context needed for microphone and camera permissions.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--config` | Built-in example | Load a local JSON configuration |
| `--checkpoint` | `models/sage-interaction` | Select model weights and model code |
| `--tts-base` | Checkpoint directory | Select separate speech assets and base TTS weights |
| `--prompt-wav` | Checkpoint's `system_ref_audio.wav` | Select the voice reference |
| `--gpu` | `0` | Comma-separated GPU indices or UUIDs |
| `--instances` | `1` | Independent workers, assigned round-robin to GPUs |
| `--vad` | `off` | Enable or disable additional VAD-based interruption detection |
| `--port` | `18080` | Browser port |
| `--worker-port` | `19310` | First worker port; later workers use consecutive ports |
| `--run-dir` | New directory under `runs/` | Store this run's output in an empty directory |
| `--ready-timeout` | `1800` | Model startup timeout in seconds |

For two independent workers on two GPUs:

```bash
bash start_demo.sh --config config.local.json --gpu 0,1 --instances 2 --vad off
```

## Time Speak

Try: **“Remind me to stretch in 60 seconds.”** Continue speaking while the reminder is pending. When the frontend delegates an explicit relative delay, the backend schedules an asynchronous timer and returns a `time_up` event at the deadline.

The current timer parser recognizes English relative delays such as `in 60 seconds` or `after 2 minutes`. The default maximum delay is 840 seconds and a session lasts at most 900 seconds. Keep that session open until the reminder fires. Closing the session cancels pending tasks. Recurring reminders and editing or cancelling individual timers are not implemented by this demo's timer parser.

## Files and run output

```text
demo/
├── start_demo.sh             # Shell entry point
├── launch.py                 # Configuration, preflight and process lifecycle
├── config.example.json       # Shareable defaults
├── requirements.txt          # Python dependencies
├── patches/                  # Reproducible speech-dependency adjustments
├── scripts/                  # Environment preparation
├── sage_demo/
│   ├── model.py              # Checkpoint-native duplex model loading
│   ├── inference.py          # Single-session voice worker
│   ├── protocol.py           # Delegation and scheduled reminders
│   ├── gateway.py            # Browser gateway and session routing
│   ├── prompts/              # Frontend and backend prompts
│   └── static/               # Browser interface
└── tests/                    # CPU-only regression checks
```

Each launch creates an independent `runs/demo-…/` directory containing configuration, logs, and local session audio and events. Session files stay on the host. `.gitignore` excludes runs, local configuration, environments, models, and credentials.

## Troubleshooting and checks

- **Missing model files:** check the checkpoint, `tts_base`, and voice-reference settings with `--check-config`.
- **Dependency mismatch:** use the pinned environment and run `python scripts/prepare_environment.py --check`.
- **GPU unavailable or out of memory:** run on the GPU host, check the selected GPU, and start with one instance.
- **Port already in use:** select an unused browser port and an unused range of worker ports.
- **Backend check fails:** verify the API key, endpoint, configured model, and host network access. The check does not submit an inference request.
- **No microphone permission:** access through localhost or an appropriately configured HTTPS deployment.
- **Worker fails to start:** inspect `runs/demo-…/logs/worker-0.log`. Startup failure stops the processes created for that run.

Run the regression checks without loading model weights:

```bash
python -m unittest discover -s tests -v
```
