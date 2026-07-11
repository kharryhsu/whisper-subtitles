"""
whisper-subtitles: Generate .srt subtitles from audio/video using faster-whisper.

Optimized for local GPU inference (e.g. NVIDIA RTX 4060) via CTranslate2.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from opencc import OpenCC
from faster_whisper import WhisperModel

def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to an SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, ms = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcribe_to_srt(
    input_path: Path,
    output_path: Path,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    traditional_chinese: bool = False,
) -> None:
    """Transcribe an audio/video file and write the result as an .srt file."""
    print(f"Loading model '{model_size}' on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    converter = OpenCC("s2twp") if traditional_chinese else None

    print(f"Transcribing: {input_path}")
    segments, info = model.transcribe(
        str(input_path),
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
    )

    print(
        f"Detected language: {info.language} "
        f"(probability {info.language_probability:.2f})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        count = 0
        for i, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            if converter:
                text = converter.convert(text)
            if not text:
                continue
            f.write(f"{i}\n")
            f.write(
                f"{format_timestamp(segment.start)} --> "
                f"{format_timestamp(segment.end)}\n"
            )
            f.write(text + "\n\n")
            count += 1

    print(f"Wrote {count} subtitle segments to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-subtitles",
        description="Generate .srt subtitles from audio/video using faster-whisper.",
    )
    parser.add_argument("input", type=Path, help="Path to audio/video file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .srt path (default: <input_stem>.srt)",
    )
    parser.add_argument(
        "-m", "--model", default="large-v3",
        help="Model size: tiny, base, small, medium, large-v3, distil-large-v3, etc. "
             "(default: large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "auto"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--compute-type", default="float16",
        help="CTranslate2 compute type, e.g. float16, int8_float16, int8 "
             "(default: float16; use int8 on lower-VRAM GPUs or CPU)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Force source language (e.g. 'th', 'zh', 'en'). "
             "Default: auto-detect.",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5, help="Beam search size (default: 5)"
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="Disable voice activity detection filtering",
    )
    parser.add_argument(
        "--traditional-chinese",
        action="store_true",
        help="Convert Chinese output to Traditional Chinese (Taiwan)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.input.with_suffix(".srt")

    transcribe_to_srt(
        input_path=args.input,
        output_path=output_path,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        traditional_chinese=args.traditional_chinese,
    )


if __name__ == "__main__":
    main()
