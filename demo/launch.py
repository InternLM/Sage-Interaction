"""Start the Sage-Interaction voice demo on the current GPU host."""

import argparse
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
ASSETS = (
    "flow.pt",
    "flow.yaml",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2_25hz.onnx",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Local JSON configuration; relative paths use its directory",
    )
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint directory")
    parser.add_argument(
        "--tts-base",
        type=Path,
        help="Base model containing Token2wav assets and optional TTS weights",
    )
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        help="Voice reference WAV; defaults to checkpoint/system_ref_audio.wav",
    )
    parser.add_argument(
        "--gpu", help="Comma-separated GPU indices or UUIDs (default: 0)"
    )
    parser.add_argument(
        "--instances",
        type=int,
        help="Independent workers distributed across the selected GPUs",
    )
    parser.add_argument(
        "--vad",
        choices=("on", "off"),
        help="Additional VAD-based interruption detection (default: off)",
    )
    parser.add_argument(
        "--port", type=int, help="Loopback browser port (default: 18080)"
    )
    parser.add_argument(
        "--worker-port", type=int, help="First loopback worker port (default: 19310)"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Empty directory for this run; defaults to a new directory under runs/",
    )
    parser.add_argument(
        "--ready-timeout", type=int, help="Startup timeout in seconds (default: 1800)"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and model files without GPU or network access",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Also check dependencies, GPUs, ports and backend access; do not load weights",
    )
    return parser.parse_args(argv)


def load_config(args):
    cfg = json.loads((ROOT / "config.example.json").read_text())
    directory = ROOT
    if args.config:
        path = args.config.expanduser().resolve(strict=True)
        values = json.loads(path.read_text())
        if not isinstance(values, dict) or set(values) - set(cfg):
            raise ValueError(
                "Configuration must contain only keys from config.example.json"
            )
        cfg.update(values)
        directory = path.parent
    for option, key in (
        ("gpu", "gpus"),
        ("instances", "instances"),
        ("port", "port"),
        ("worker_port", "worker_port"),
        ("ready_timeout", "ready_timeout"),
    ):
        if getattr(args, option) is not None:
            cfg[key] = getattr(args, option)
    if args.vad is not None:
        cfg["enable_system_barge_in"] = args.vad == "on"

    def resolve(key, fallback=None):
        override = getattr(args, key, None)
        if override is not None:
            return override.expanduser().resolve(strict=True)
        value = cfg.get(key) or fallback
        if value is None:
            raise ValueError("Missing configuration: " + key)
        return (directory / Path(value).expanduser()).resolve(strict=True)

    checkpoint = resolve("checkpoint")
    tts = resolve("tts_base", checkpoint)
    prompt = resolve("prompt_wav", checkpoint / "system_ref_audio.wav")
    cfg.update(
        frontend_model=str(checkpoint),
        tts_base_model=str(tts),
        prompt_wav=str(prompt),
        frontend_model_name="Sage-Interaction",
        worker_count=cfg["instances"],
    )
    for key in ("backend_prompt", "frontend_prompt_file"):
        # Prompts shipped with the demo are package resources, not machine configuration.
        cfg[key] = str((ROOT / cfg[key]).resolve(strict=True))
    if (
        not isinstance(cfg["instances"], int)
        or isinstance(cfg["instances"], bool)
        or cfg["instances"] < 1
    ):
        raise ValueError("instances must be a positive integer")
    gpus = [value.strip() for value in str(cfg["gpus"]).split(",")]
    if any(
        not re.fullmatch(r"[0-9]+|GPU-[A-Fa-f0-9-]+", value) for value in gpus
    ) or len(set(gpus)) != len(gpus):
        raise ValueError("gpu must list distinct GPU indices or UUIDs")
    cfg["gpu_devices"] = gpus
    ports = [
        cfg["port"],
        *range(cfg["worker_port"], cfg["worker_port"] + cfg["instances"]),
    ]
    if len(set(ports)) != len(ports) or any(
        not 1024 <= port <= 65535 for port in ports
    ):
        raise ValueError("Use distinct ports between 1024 and 65535")
    if cfg["ready_timeout"] <= 0:
        raise ValueError("ready_timeout must be positive")
    if cfg["chunk_samples"] != 16000 or cfg["frontend_precision"] != "bfloat16":
        raise ValueError(
            "This demo uses one-second 16 kHz input chunks and native BF16 inference"
        )
    if cfg["max_reminder_seconds"] >= cfg["max_session_seconds"]:
        raise ValueError(
            "max_reminder_seconds must be shorter than max_session_seconds"
        )
    cfg["backend_base_url"] = os.environ.get(
        "SAGE_BACKEND_BASE_URL", cfg["backend_base_url"]
    ).rstrip("/")
    cfg["backend_model_name"] = os.environ.get(
        "SAGE_BACKEND_MODEL", cfg["backend_model_name"]
    )
    url = urllib.parse.urlsplit(cfg["backend_base_url"])
    if (
        url.scheme not in {"https", "http"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "backend_base_url must be an HTTP(S) API base URL without credentials or query parameters"
        )
    if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use HTTPS for a remote backend")
    return cfg


def validate_model_files(cfg):
    checkpoint, tts = Path(cfg["frontend_model"]), Path(cfg["tts_base_model"])
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    required = [checkpoint / name for name in set(weights.values())]
    required += [
        checkpoint / "config.json",
        checkpoint / "modeling_minicpmo.py",
        Path(cfg["prompt_wav"]),
    ]
    required += [tts / "assets/token2wav" / name for name in ASSETS]
    if not any(key.startswith("tts.") for key in weights):
        base = json.loads((tts / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        shards = {shard for key, shard in base.items() if key.startswith("tts.")}
        if not shards:
            raise ValueError("The checkpoint and TTS base both lack TTS weights")
        required += [tts / name for name in shards]
    if any(not path.is_file() for path in required):
        missing = sorted({path.name for path in required if not path.is_file()})
        raise ValueError("Missing model assets: " + ", ".join(missing))


def check_environment(cfg):
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Python 3.12 or later is required; Python 3.12 is the tested version"
        )
    for module in (
        "torch",
        "transformers",
        "accelerate",
        "stepaudio2",
        "fastapi",
        "uvicorn",
        "httpx",
        "numpy",
        "scipy",
        "soundfile",
        "websockets",
        "onnxruntime",
        "webrtcvad",
        "silero_vad",
        "flash_attn",
    ):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(
                "Missing dependency: " + module + "; install requirements.txt first"
            )
    from scripts.prepare_environment import prepare

    prepare(check_only=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["gpu_devices"])
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != len(
        cfg["gpu_devices"]
    ):
        raise RuntimeError("The selected GPUs are unavailable on this host")
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Every selected GPU must support BF16")
    spec = importlib.util.find_spec("silero_vad")
    model = Path(next(iter(spec.submodule_search_locations))) / "data/silero_vad.onnx"
    if cfg["enable_input_noise_filter"] and not model.is_file():
        raise RuntimeError("The installed Silero VAD package is missing its ONNX model")
    cfg["input_noise_speech_model"] = str(model)


def check_ports(cfg):
    for port in [
        cfg["port"],
        *range(cfg["worker_port"], cfg["worker_port"] + cfg["instances"]),
    ]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def check_backend(cfg):
    from sage_demo.protocol import backend_headers

    key = os.environ.get("SAGE_BACKEND_API_KEY", "").strip()
    if not key:
        if not sys.stdin.isatty():
            raise ValueError("Set SAGE_BACKEND_API_KEY before non-interactive startup")
        key = getpass.getpass("Backend API key (hidden): ").strip()
        os.environ["SAGE_BACKEND_API_KEY"] = key
    request = urllib.request.Request(
        cfg["backend_base_url"] + "/models", headers=backend_headers()
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            models = json.load(response)
        if cfg["backend_model_name"] not in [row["id"] for row in models["data"]]:
            raise RuntimeError("The backend does not list the configured model")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Backend access check failed (HTTP {error.code})") from None
    except (OSError, ValueError, KeyError):
        raise RuntimeError(
            "Backend access check failed; verify the API endpoint and network"
        ) from None


def stop_children(children):
    """Terminate only process groups created by this launcher."""
    for child in reversed(children):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for child in reversed(children):
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def serve(cfg, args):
    os.umask(0o077)
    if args.run_dir:
        run = args.run_dir.expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
        if any(run.iterdir()):
            raise ValueError("run-dir must be empty")
        run.chmod(0o700)
    else:
        (ROOT / "runs").mkdir(exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="demo-", dir=ROOT / "runs"))
    (run / "logs").mkdir()
    cfg["run_directory"] = str(run)
    config_path = run / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    config_path.chmod(0o600)
    children, logs = [], []

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(run / "cache/torch")
    try:
        for index in range(cfg["instances"]):
            log = (run / "logs" / f"worker-{index}.log").open("w")
            logs.append(log)
            env = dict(
                environment,
                CUDA_VISIBLE_DEVICES=cfg["gpu_devices"][
                    index % len(cfg["gpu_devices"])
                ],
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "sage_demo.inference",
                        "--config",
                        str(config_path),
                        "--device",
                        str(index),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            )
        log = (run / "logs/gateway.log").open("w")
        logs.append(log)
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sage_demo.gateway",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        )
        print(
            f"Loading {cfg['instances']} independent worker(s). Logs: {run.name}/logs",
            flush=True,
        )
        url = f"http://127.0.0.1:{cfg['port']}/"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + cfg["ready_timeout"]
        while time.monotonic() < deadline:
            if any(child.poll() is not None for child in children):
                raise RuntimeError(
                    "A demo process exited during startup; inspect this run's logs"
                )
            try:
                with opener.open(url + "api/health", timeout=3) as response:
                    health = json.load(response)
                workers = health["workers"]
                if len(workers) == cfg["instances"] and all(
                    row["ready"] for row in workers
                ):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Model startup timed out; inspect this run's logs")
        print(f"Demo ready: {url}\nPress Ctrl+C to stop this run.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError("A demo process exited; inspect this run's logs")
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_children(children)
        finally:
            for log in logs:
                log.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args)
    validate_model_files(cfg)
    if args.check_config:
        print(
            "Configuration and model files verified; VAD interruption: "
            + ("on" if cfg["enable_system_barge_in"] else "off")
            + "; scheduled reminders: "
            + ("on" if cfg["enable_scheduled_reminders"] else "off")
        )
        return
    check_environment(cfg)
    check_ports(cfg)
    check_backend(cfg)
    if args.check_only:
        print("Preflight passed; no model weights were loaded")
        return
    serve(cfg, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Demo stopped")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print("Startup failed: " + str(error), file=sys.stderr)
        sys.exit(1)
