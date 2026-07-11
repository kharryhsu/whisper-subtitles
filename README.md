# whisper-subtitles

Generate `.srt` subtitles from audio or video files using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with an optional second stage that translates the result into a bilingual `.srt` using a local NLLB-200 model.

## Features

- Stage 1 (`transcribe.py`): audio/video → `.srt`, tuned for mixed-language (e.g. Thai/Chinese code-switched) source audio
- Stage 2 (`translate.py`): source `.srt` → bilingual `.srt`, offline via NLLB-200

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
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Stage 2 (translation) needs a couple of extra packages not required for transcription alone:

```bash
pip install ctranslate2 transformers sentencepiece
```

## Usage

### Stage 1: Transcribe

```bash
whisper-subtitles video.mp4
```

This writes `video.srt` next to the input file, auto-detecting the spoken language(s).

#### Common options

```bash
# Custom output path
whisper-subtitles video.mp4 -o subs/my_video.srt

# Smaller/faster model (useful for quick drafts or lower VRAM)
whisper-subtitles video.mp4 -m medium

# Force a language instead of auto-detecting
whisper-subtitles video.mp4 --language th

# Lower VRAM usage via int8 quantization (default)
whisper-subtitles video.mp4 --compute-type int8_float16

# Full float16 (more accurate, needs more VRAM headroom)
whisper-subtitles video.mp4 --compute-type float16

# CPU-only (no CUDA GPU available)
whisper-subtitles video.mp4 --device cpu --compute-type int8
```

> On an RTX 4060 (8 GB), `--compute-type float16` with `large-v3` fits fine in practice despite the ~10 GB nominal figure below — actual usage depends on what else is holding VRAM. `int8_float16` remains the safe default if you hit an out-of-memory error.

#### Mixed-language audio

The defaults are already tuned for this case:

- `--language` is left unset so the model auto-detects language per chunk instead of committing to one language for the whole file.
- `--vad-min-silence-ms` defaults to `300`, which splits on pauses more often, giving the model more chances to re-detect language when it switches mid-video.
- `condition_on_previous_text` is off by default, since otherwise the model biases toward repeating the previous segment's language even after the speaker switches.

If a whole segment still comes out in the wrong language, try lowering `--vad-min-silence-ms` further (e.g. `200`) to force smaller segments.

Run `whisper-subtitles --help` for the full option list.

#### Model sizes

| Model            | Relative speed | VRAM (approx, float16) | Notes                          |
|-------------------|---------------|--------------------------|---------------------------------|
| `tiny` / `base`   | fastest       | ~1 GB                    | Draft quality                   |
| `small`           | fast          | ~2 GB                    | Decent for clean audio          |
| `medium`          | moderate      | ~5 GB                    | Good balance                    |
| `large-v3`        | slower        | ~10 GB (nominal)         | Best accuracy (default)         |
| `distil-large-v3` | faster        | ~6 GB                    | Near large-v3 accuracy, lighter |

An RTX 4060 (8 GB) comfortably runs `medium` or `distil-large-v3` in `float16`. For `large-v3`, try `float16` first — it may fit depending on available VRAM — and fall back to `--compute-type int8_float16` (the default) if you hit an out-of-memory error.

### Stage 2: Translate into a bilingual subtitle

Once you have a source `.srt` from Stage 1, translate it into a second language. Each cue in the output shows the original line followed by its translation, stacked in the same timestamp block.

#### Choosing an NLLB model size

| Model | Disk size | VRAM (int8) | Notes |
|---|---|---|---|
| `distilled-600M` | ~2.5GB | ~1GB | Fastest, but noticeably weaker — can produce nonsense translations, especially for lower-resource pairs like Thai |
| `distilled-1.3B` | ~5.5GB | ~2GB | Good balance of speed and quality |
| `3.3B` | ~13GB | ~4GB | Best quality, still fits an 8GB card in int8, slower per line |

NLLB is released under CC-BY-NC 4.0 — fine for personal/non-commercial use, but worth knowing if you ever repurpose this pipeline commercially.

Get an NLLB-200 model converted to CTranslate2 format (one-time setup per model size):

```bash
pip install transformers[sentencepiece]

# 600M (fastest, lower quality)
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-600M-ct2 --quantization int8

# 1.3B (balanced)
ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B --output_dir nllb-200-1.3B-ct2 --quantization int8

# 3.3B (best quality)
ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir nllb-200-3.3B-ct2 --quantization int8
```

Then run the translation, passing `--tokenizer-name` to match whichever model repo you converted (the tokenizer is downloaded separately and cached, since `ct2-transformers-converter` doesn't copy tokenizer files into the output directory):

```bash
whisper-subtitles-translate video.srt \
  --model-dir nllb-200-3.3B-ct2 \
  --tokenizer-name facebook/nllb-200-3.3B \
  --target-lang en
```

Output:

```
1
00:00:01,200 --> 00:00:03,400
สวัสดีครับ
Hello
```

#### Common options

```bash
# Translate to Traditional Chinese instead of Simplified
whisper-subtitles-translate video.srt --model-dir nllb-200-3.3B-ct2 --tokenizer-name facebook/nllb-200-3.3B --target-lang zh --traditional-chinese

# Force a single source language instead of auto-detecting per line
whisper-subtitles-translate video.srt --model-dir nllb-200-3.3B-ct2 --tokenizer-name facebook/nllb-200-3.3B --target-lang en --source-lang th

# Raise beam size if translations are still not accurate enough (default: 5)
whisper-subtitles-translate video.srt --model-dir nllb-200-3.3B-ct2 --tokenizer-name facebook/nllb-200-3.3B --target-lang en --beam-size 8

# CPU-only
whisper-subtitles-translate video.srt --model-dir nllb-200-3.3B-ct2 --tokenizer-name facebook/nllb-200-3.3B --target-lang en --device cpu
```

For mixed Thai/Chinese source subtitles, source language is auto-detected per cue by Unicode block (Thai vs Han), so each line translates from whatever language it's actually written in. Supported languages: `th`, `zh`, `en`, `ja`.

Run `whisper-subtitles-translate --help` for the full option list.

## Project structure

```
whisper-subtitles/
├── .github/workflows/
│   └── lint.yml            # ruff CI check (runs on Python 3.11)
├── src/whisper_subtitles/
│   ├── __init__.py
│   ├── transcribe.py      # Stage 1: audio/video -> .srt (faster-whisper)
│   └── translate.py       # Stage 2: .srt -> bilingual .srt (NLLB-200)
├── .gitignore
├── .python-version         # pins Python 3.11 for pyenv/asdf/etc.
├── pyproject.toml
├── requirements-lock.txt         # exact locked dependency versions
├── LICENSE
└── README.md
```

## Troubleshooting

- **`Could not load library libcudnn_ops_infer.so`**: your CUDA/cuDNN version doesn't match what `ctranslate2` expects. Check the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) for the exact versions, or fall back to `--device cpu`.
- **Out of memory (Stage 1)**: switch to a smaller model (`-m medium`), or use a quantized compute type (`--compute-type int8_float16` or `int8`).
- **Wrong language detected on short clips**: pass `--language` explicitly, since auto-detection is less reliable on very short audio.
- **`Library cublas64_12.dll is not found or cannot be loaded`**: usually means `nvidia-cublas-cu12` and/or `nvidia-cudnn-cu12` aren't actually installed in this venv (check with `pip show nvidia-cublas-cu12 nvidia-cudnn-cu12`; install with `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` if missing). Once installed, add both bin folders to `PATH`:
    ```powershell
    $env:PATH += ";$PWD\.venv\Lib\site-packages\nvidia\cublas\bin;$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin"
    ```
    For a permanent fix, add this to your virtual environment's `Activate.ps1` so it's set automatically on every `.venv` activation, rather than adding it to the global Windows `PATH` (keeps it scoped to this project/venv).
- **Stage 2 translations look off for short/ambiguous lines**: NLLB relies on the detected source language; if a cue is very short (e.g. a single word) the Unicode-block detector can still work, but very short or code-mixed single lines may translate awkwardly regardless. Consider `--source-lang` to force a language if you know the whole file is dominated by one.
- **Stage 2 translations are nonsensical, not just awkward**: check that `--beam-size` isn't set to `1` (default is `5`, greedy decoding produces much worse output). If it's still bad at the default beam size, the model itself may be too small — try `distilled-1.3B` or `3.3B` instead of `distilled-600M`.
- **Stage 2 tokenizer error (`Couldn't instantiate the backend tokenizer...`)**: make sure `--tokenizer-name` matches the exact model repo you converted (e.g. `facebook/nllb-200-3.3B` if you converted the 3.3B model). Passing `--model-dir` alone as the tokenizer source won't work — `ct2-transformers-converter` only converts the model weights, not the tokenizer files.

## Notes

- The converted NLLB model directories (e.g. `nllb-200-600M-ct2/`, `nllb-200-1.3B-ct2/`, `nllb-200-3.3B-ct2/`) are build artifacts, not source — add them to `.gitignore` rather than committing them. Anyone cloning the repo can regenerate whichever size they need with the `ct2-transformers-converter` commands above. A pattern like `nllb-*-ct2/` in `.gitignore` covers all sizes at once.

## Roadmap

- Optional single command that chains Stage 1 → Stage 2 automatically (currently two separate CLI calls: `whisper-subtitles` then `whisper-subtitles-translate`)
- Optional bilingual `.ass`/styled subtitle output (separate font/position per language line)

## License

[MIT](LICENSE)