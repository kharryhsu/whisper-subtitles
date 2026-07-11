# whisper-subtitles

Generate `.srt` subtitles from audio or video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Features

-

## Requirements

- Python 3.9–3.12 (3.11 recommended)
- NVIDIA GPU with CUDA support (CPU fallback available, but much slower)

## Installation

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/whisper-subtitles.git
cd whisper-subtitles
```

If you use [pyenv](https://github.com/pyenv/pyenv) or a version manager that reads `.python-version`, it'll pick up 3.11 automatically. Otherwise, point explicitly at a 3.9–3.12 interpreter when creating the venv:

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```bash
whisper-subtitles video.mp4
```

This writes `video.srt` next to the input file, auto-detecting the spoken language(s).

### Common options

```bash
# Custom output path
whisper-subtitles video.mp4 -o subs/my_video.srt

# Smaller/faster model (useful for quick drafts or lower VRAM)
whisper-subtitles video.mp4 -m medium

# Force a language instead of auto-detecting
whisper-subtitles video.mp4 --language th

# Lower VRAM usage via int8 quantization
whisper-subtitles video.mp4 --compute-type int8_float16

# CPU-only (no CUDA GPU available)
whisper-subtitles video.mp4 --device cpu --compute-type int8
```

Run `whisper-subtitles --help` for the full option list.

### Model sizes

| Model            | Relative speed | VRAM (approx, float16) | Notes                          |
|-------------------|---------------|--------------------------|---------------------------------|
| `tiny` / `base`   | fastest       | ~1 GB                    | Draft quality                   |
| `small`           | fast          | ~2 GB                    | Decent for clean audio          |
| `medium`          | moderate      | ~5 GB                    | Good balance                    |
| `large-v3`        | slower        | ~10 GB                   | Best accuracy (default)         |
| `distil-large-v3` | faster        | ~6 GB                    | Near large-v3 accuracy, lighter |

An RTX 4060 (8 GB) comfortably runs `medium` or `distil-large-v3` in `float16`; for `large-v3`, use `--compute-type int8_float16` to fit in 8 GB VRAM.

## Example output

- 

## Project structure

```
whisper-subtitles/
├── .github/workflows/
│   └── lint.yml            # ruff CI check (runs on Python 3.11)
├── src/whisper_subtitles/
│   ├── __init__.py
│   └── transcribe.py      # core transcription logic + CLI
├── .gitignore
├── .python-version         # pins Python 3.11 for pyenv/asdf/etc.
├── pyproject.toml
├── requirements.txt         # exact locked dependency versions
├── LICENSE
└── README.md
```

## Troubleshooting

- **`Could not load library libcudnn_ops_infer.so`**: your CUDA/cuDNN version doesn't match what `ctranslate2` expects. Check the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) for the exact versions, or fall back to `--device cpu`.
- **Out of memory**: switch to a smaller model (`-m medium`), or use a quantized compute type (`--compute-type int8_float16` or `int8`).
- **Wrong language detected on short clips**: pass `--language` explicitly, since auto-detection is less reliable on very short audio.
- **`Library cublas64_12.dll is not found or cannot be loaded`**: `faster-whisper`/`ctranslate2` cannot find the required CUDA runtime libraries. Make sure the NVIDIA CUDA libraries are installed and available in your system `PATH`. If you installed CUDA packages through pip, add the cuBLAS binary folder: 
    ```powershell
    $env:PATH += ";$PWD\.venv\Lib\site-packages\nvidia\cublas\bin"
    ```
    For a permanent fix, add this folder to your Windows Environment Variables or your virtual environment activation script.

## Roadmap

- Translate the missing language per segment using a local LLM endpoint
- Merge into a bilingual `.srt`

## License

[MIT](LICENSE)