"""
whisper-subtitles translate: Generate bilingual .srt subtitles by translating
an existing .srt with a local NLLB-200 model (via CTranslate2).

Designed as Stage 2 of the pipeline: run transcribe.py first to get a source
.srt, then run this to produce a bilingual .srt where each cue shows the
original line followed by its translation.

Offline, no API calls. Handles per-line language detection for mixed
Thai/Chinese source subtitles (auto mode), or you can force a single source
language.
"""

from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer
from opencc import OpenCC


# FLORES-200 language codes used by NLLB
LANG_CODES = {
    "th": "tha_Thai",
    "zh": "zho_Hans",       # Simplified Chinese
    "zh-tw": "zho_Hant",    # Traditional Chinese (direct NLLB output)
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
}

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
HAN_RE = re.compile(r"[\u4E00-\u9FFF]")


@dataclass
class SrtCue:
    index: int
    start: str
    end: str
    text: str


def parse_srt(path: Path) -> list[SrtCue]:
    """Parse an .srt file into a list of SrtCue objects."""
    raw = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        start, end = time_match.group(1), time_match.group(2)
        text = " ".join(lines[2:]).strip()
        if text:
            cues.append(SrtCue(index=index, start=start, end=end, text=text))
    return cues


def detect_lang(text: str) -> str:
    """Detect Thai vs Chinese by Unicode block. Falls back to 'en'."""
    if THAI_RE.search(text):
        return "th"
    if HAN_RE.search(text):
        return "zh"
    return "en"


class NllbTranslator:
    """Thin wrapper around a CTranslate2-converted NLLB-200 model."""

    def __init__(
        self,
        model_dir: str,
        tokenizer_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda",
        compute_type: str = "int8",
    ):
        # model_dir holds the CTranslate2-converted weights only.
        # ct2-transformers-converter does not copy tokenizer files, so the
        # tokenizer is loaded separately from the original HF repo (downloads
        # a few MB once, then caches locally — no relation to the CT2 weights).
        self.translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def translate(self, text: str, src_lang: str, tgt_lang: str, beam_size: int = 5) -> str:
        src_code = LANG_CODES[src_lang]
        tgt_code = LANG_CODES[tgt_lang]
        self.tokenizer.src_lang = src_code
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.translator.translate_batch(
            [tokens],
            target_prefix=[[tgt_code]],
            beam_size=beam_size,
        )
        out_tokens = results[0].hypotheses[0][1:]  # drop the forced target-lang token
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def build_bilingual_srt(
    input_path: Path,
    output_path: Path,
    model_dir: str,
    target_lang: str,
    source_lang: str | None = None,
    tokenizer_name: str = "facebook/nllb-200-distilled-600M",
    device: str = "cuda",
    compute_type: str = "int8",
    beam_size: int = 5,
    traditional_chinese: bool = False,
) -> None:
    cues = parse_srt(input_path)
    if not cues:
        print(f"No cues found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading NLLB model from '{model_dir}' on {device} ({compute_type})...")
    translator = NllbTranslator(model_dir, tokenizer_name=tokenizer_name, device=device, compute_type=compute_type)
    converter = OpenCC("s2twp") if (traditional_chinese and target_lang == "zh") else None
    effective_target = "zh-tw" if (traditional_chinese and target_lang == "zh") else target_lang

    print(f"Translating {len(cues)} cues -> {effective_target}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, start=1):
            src_lang = source_lang or detect_lang(cue.text)
            if src_lang == effective_target.split("-")[0]:
                # Source and target are already the same language, skip translating
                translated = cue.text
            else:
                translated = translator.translate(cue.text, src_lang, effective_target, beam_size=beam_size)
                if converter:
                    translated = converter.convert(translated)

            f.write(f"{i}\n")
            f.write(f"{cue.start} --> {cue.end}\n")
            f.write(f"{cue.text}\n{translated}\n\n")

            if i % 25 == 0 or i == len(cues):
                print(f"  {i}/{len(cues)}")

    print(f"Wrote bilingual subtitles to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-subtitles-translate",
        description="Translate an .srt into a bilingual .srt using a local NLLB-200 model.",
    )
    parser.add_argument("input", type=Path, help="Path to source .srt file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output bilingual .srt path (default: <input_stem>.bilingual.srt)",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the CTranslate2-converted NLLB-200 model directory",
    )
    parser.add_argument(
        "--tokenizer-name", default="facebook/nllb-200-distilled-600M",
        help="Hugging Face repo id to load the tokenizer from (default: "
             "facebook/nllb-200-distilled-600M). This must match whichever "
             "NLLB variant you converted for --model-dir. Downloaded and "
             "cached automatically; unrelated to the CT2 weights folder.",
    )
    parser.add_argument(
        "--target-lang", required=True, choices=["th", "zh", "en", "ja"],
        help="Language to translate into",
    )
    parser.add_argument(
        "--source-lang", default=None, choices=["th", "zh", "en", "ja"],
        help="Force source language for every cue. Default: auto-detect per line "
             "(recommended for mixed Thai/Chinese source subtitles).",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu", "auto"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--compute-type", default="int8",
        help="CTranslate2 compute type (default: int8)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5,
        help="Beam search size for translation decoding (default: 5). "
             "The previous default was effectively 1 (greedy), which produces "
             "noticeably worse, sometimes nonsensical translations. Higher "
             "values are slower but more accurate; try 8-10 if quality is "
             "still not good enough.",
    )
    parser.add_argument(
        "--traditional-chinese", action="store_true",
        help="If target is 'zh', output Traditional Chinese instead of Simplified",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.input.with_suffix(".bilingual.srt")

    build_bilingual_srt(
        input_path=args.input,
        output_path=output_path,
        model_dir=args.model_dir,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        traditional_chinese=args.traditional_chinese,
    )


if __name__ == "__main__":
    main()
