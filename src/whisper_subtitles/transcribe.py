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


def split_into_chunks(words, max_duration: float, max_chars: int):
    """
    Split a list of word objects (with .start, .end, .word) into smaller
    chunks, each capped by max_duration (seconds) and max_chars (characters).
    Returns a list of (start, end, text) tuples.
    """
    chunks = []
    cur_words = []
    cur_start = None

    def flush():
        if not cur_words:
            return
        start = cur_words[0].start
        end = cur_words[-1].end
        text = "".join(w.word for w in cur_words).strip()
        if text:
            chunks.append((start, end, text))

    for w in words:
        if cur_start is None:
            cur_start = w.start

        candidate_text = "".join(x.word for x in cur_words) + w.word
        candidate_duration = w.end - cur_start

        if cur_words and (
            candidate_duration > max_duration or len(candidate_text.strip()) > max_chars
        ):
            flush()
            cur_words = [w]
            cur_start = w.start
        else:
            cur_words.append(w)

    flush()
    return chunks


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
    max_duration: float = 6.0,
    max_chars: int = 35,
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
        word_timestamps=True,
    )

    print(
        f"Detected language: {info.language} "
        f"(probability {info.language_probability:.2f})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        count = 0
        for segment in segments:
            words = segment.words or []
            if not words:
                text = segment.text.strip()
                if converter:
                    text = converter.convert(text)
                if not text:
                    continue
                count += 1
                f.write(f"{count}\n")
                f.write(
                    f"{format_timestamp(segment.start)} --> "
                    f"{format_timestamp(segment.end)}\n"
                )
                f.write(text + "\n\n")
                continue

            for start, end, text in split_into_chunks(words, max_duration, max_chars):
                if converter:
                    text = converter.convert(text)
                if not text:
                    continue
                count += 1
                f.write(f"{count}\n")
                f.write(
                    f"{format_timestamp(start)} --> "
                    f"{format_timestamp(end)}\n"
                )
                f.write(text + "\n\n")

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
    parser.add_argument(
        "--max-duration", type=float, default=6.0,
        help="Max seconds per subtitle chunk (default: 6.0)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=35,
        help="Max characters per subtitle chunk (default: 35)",
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
        max_duration=args.max_duration,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
