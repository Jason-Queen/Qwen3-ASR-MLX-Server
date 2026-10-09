"""Validated, ASR-independent contracts for authoritative transcript alignment."""
from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

MAX_WINDOW_SECONDS = 30.0
MAX_WINDOW_UNITS = 256
MAX_WINDOW_TEXT_TOKENS = 2048
MAX_TEXT_CHARACTERS = 100_000
MAX_REGION_JSON_BYTES = 1024 * 1024
MAX_REGIONS = 1000
MAX_SOURCE_SECONDS = 6 * 60 * 60
SAMPLE_RATE = 16000
TIMESTAMP_RESOLUTION_SECONDS = 0.08
RESPONSE_FORMATS = frozenset({"verbose_json", "srt", "vtt"})


class AlignmentInputError(ValueError):
    """Invalid input or a documented operating limit was exceeded."""


class AlignmentOutputError(RuntimeError):
    """Model output cannot be verified without changing the transcript/timing."""


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    text_start: int
    text_end: int


@dataclass(frozen=True)
class AlignmentRequest:
    text: str
    language: str
    response_format: str
    regions: tuple[Region, ...] | None


@dataclass(frozen=True)
class Window:
    start_frame: int
    end_frame: int
    text_start: int
    text_end: int

    @property
    def start(self) -> float:
        return self.start_frame / SAMPLE_RATE

    @property
    def end(self) -> float:
        return self.end_frame / SAMPLE_RATE


@dataclass
class AlignmentResult:
    text: str
    language: str
    duration: float
    words: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "task": "align",
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "transcript_source": "supplied",
            "timestamp_provenance": {
                "words": "forced_alignment",
                "segments": "forced_alignment",
                "alignment": "applied_with_warnings" if (
                    self.diagnostics["zero_duration_word_indices"]
                    or self.diagnostics["boundary_adjustments"]
                    or self.diagnostics["overlapping_word_indices"]
                ) else "applied",
                "timestamp_repair": "none",
            },
            "words": self.words,
            "segments": self.segments,
            "regions": self.regions,
            "diagnostics": self.diagnostics,
        }


def _finite_number(value: Any, label: str, error: type[ValueError] | type[RuntimeError]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise error(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise error(f"{label} must be a finite number")
    return number


def _unique_region_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("Duplicate region keys")
    return values


def parse_request(text: Any, language: Any, response_format: Any, regions: Any = None) -> AlignmentRequest:
    if not isinstance(text, str) or not text.strip():
        raise AlignmentInputError("text is required and must contain an authoritative transcript")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise AlignmentInputError(f"text exceeds {MAX_TEXT_CHARACTERS} Unicode code points")
    if any(unicodedata.category(c) in {"Cs", "Cc"} and c not in "\t\r\n" for c in text):
        raise AlignmentInputError("text contains invalid Unicode or unsupported control characters")
    if not isinstance(language, str) or not language.strip() or language.strip().lower() == "auto":
        raise AlignmentInputError("language is required; automatic ASR language detection is not used")
    if not isinstance(response_format, str) or response_format not in RESPONSE_FORMATS:
        raise AlignmentInputError("Alignment response_format must be verbose_json, srt, or vtt")
    if response_format != "verbose_json":
        _validate_subtitle_text(text)
    if regions is None:
        return AlignmentRequest(text, language.strip(), response_format, None)
    if isinstance(regions, str):
        if len(regions.encode("utf-8")) > MAX_REGION_JSON_BYTES:
            raise AlignmentInputError("regions JSON exceeds 1 MiB")
        try:
            regions = json.loads(regions, object_pairs_hook=_unique_region_keys)
        except (ValueError, RecursionError):
            raise AlignmentInputError("regions must be a JSON array of region objects with unique keys") from None
    if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
        raise AlignmentInputError(f"regions must be a nonempty array with at most {MAX_REGIONS} entries")
    parsed: list[Region] = []
    text_cursor = 0
    previous_end = 0.0
    for index, region in enumerate(regions):
        label = f"regions[{index}]"
        if not isinstance(region, dict) or set(region) != {"start", "end", "text_start", "text_end"}:
            raise AlignmentInputError(f"{label} requires exactly start, end, text_start, text_end")
        start = _finite_number(region["start"], f"{label}.start", AlignmentInputError)
        end = _finite_number(region["end"], f"{label}.end", AlignmentInputError)
        if start < previous_end or end <= start:
            raise AlignmentInputError(f"{label} audio range must be positive, ordered, nonnegative and non-overlapping")
        first, last = region["text_start"], region["text_end"]
        if type(first) is not int or type(last) is not int or first != text_cursor or not first < last <= len(text):
            raise AlignmentInputError(f"{label} text offsets must give ordered contiguous coverage of the full text")
        if end - start > MAX_WINDOW_SECONDS + 1e-9:
            raise AlignmentInputError(f"{label} exceeds {MAX_WINDOW_SECONDS:g} seconds; supply smaller explicit regions with known text boundaries")
        parsed.append(Region(start, end, first, last))
        text_cursor, previous_end = last, end
    if text_cursor != len(text):
        raise AlignmentInputError("regions must cover the complete text without omissions or duplication")
    return AlignmentRequest(text, language.strip(), response_format, tuple(parsed))


def plan_windows(request: AlignmentRequest, frame_count: int) -> list[Window]:
    duration = frame_count / SAMPLE_RATE
    if frame_count <= 0:
        raise AlignmentInputError("Audio contains no samples")
    if duration > MAX_SOURCE_SECONDS:
        raise AlignmentInputError("Decoded source audio exceeds the six-hour resource limit")
    if request.regions is None:
        if duration > MAX_WINDOW_SECONDS:
            raise AlignmentInputError(
                f"Unanchored alignment supports at most {MAX_WINDOW_SECONDS:g} seconds; "
                "provide regions (HTTP) or --regions-file (CLI) with known audio/text boundaries. "
                "No ASR or proportional text splitting is performed."
            )
        return [Window(0, frame_count, 0, len(request.text))]
    windows: list[Window] = []
    for index, region in enumerate(request.regions):
        # Nearest sample; at most half a sample of rounding at either endpoint.
        if region.end > duration + 0.5 / SAMPLE_RATE:
            raise AlignmentInputError(f"regions[{index}].end exceeds source duration {duration:.6f}s")
        start = int(math.floor(region.start * SAMPLE_RATE + 0.5))
        end = int(math.floor(region.end * SAMPLE_RATE + 0.5))
        end = min(end, frame_count)
        if end <= start or (windows and start < windows[-1].end_frame):
            raise AlignmentInputError(f"regions[{index}] is empty or overlapping after sample rounding")
        if end - start > int(MAX_WINDOW_SECONDS * SAMPLE_RATE):
            raise AlignmentInputError(f"regions[{index}] exceeds the per-window sample limit")
        windows.append(Window(start, end, region.text_start, region.text_end))
    return windows


def _kept(character: str) -> bool:
    # Match the installed Qwen ForceAlignProcessor without changing source text.
    return character == "'" or unicodedata.category(character)[0] in {"L", "N"}


def source_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    positions = [index for index, character in enumerate(text) if _kept(character)]
    identity = "".join(text[index] for index in positions)
    if not tokens or any(not token for token in tokens):
        raise AlignmentInputError("Transcript or region contains no alignable tokens")
    if "".join(tokens) != identity:
        raise AlignmentInputError("Canonical tokens cannot be mapped exactly to supplied Unicode text; revise the region boundary or unsupported tokenizer normalization")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        end = cursor + len(token)
        source_end = positions[end - 1] + 1
        while source_end < len(text) and unicodedata.category(text[source_end]).startswith("M"):
            source_end += 1
        spans.append((positions[cursor], source_end))
        cursor = end
    return spans


def validate_words(raw: Any, tokens: list[str], duration: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list) or [w.get("word") if isinstance(w, dict) else None for w in raw] != tokens:
        raise AlignmentOutputError("Forced aligner token identity/coverage mismatch")
    words: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    previous_start = previous_end = 0.0
    for index, item in enumerate(raw):
        start = _finite_number(item.get("start"), f"word[{index}].start", AlignmentOutputError)
        end = _finite_number(item.get("end"), f"word[{index}].end", AlignmentOutputError)
        if (start < 0 or end < start or start < previous_start or end < previous_end
                or previous_end - start > TIMESTAMP_RESOLUTION_SECONDS + 1 / SAMPLE_RATE):
            raise AlignmentOutputError(f"Invalid, backward, or excessively overlapping forced-alignment timestamp at word[{index}]; no timestamp repair was applied")
        previous_start, previous_end = start, end
        for field, value in (("start", start), ("end", end)):
            if value > duration:
                if value - duration > TIMESTAMP_RESOLUTION_SECONDS + 1 / SAMPLE_RATE:
                    raise AlignmentOutputError(f"Forced-alignment word[{index}].{field} exceeds its audio window")
                adjustments.append({"word_index": index, "field": field, "original": value, "adjusted": duration})
        # Only the documented <= one-resolution boundary overshoot is adjusted.
        start, end = min(start, duration), min(end, duration)
        words.append({"word": item["word"], "start": start, "end": end,
                      "timing_status": "zero_duration" if start == end else "aligned"})
    return words, adjustments


def build_segments(text: str, words: list[dict[str, Any]], sentence_ends: list[int], region_index: int) -> list[dict[str, Any]]:
    """Partition source slices, never reconstruct text by joining model tokens."""
    boundaries: dict[int, int] = {}
    for end in sentence_ends:
        count = sum(word["text_end"] <= end for word in words)
        if 0 < count < len(words) and words[count]["text_start"] >= end:
            # Include whitespace after sentence punctuation in its exact slice.
            while end < len(text) and text[end].isspace():
                end += 1
            boundaries[count] = end
    segments: list[dict[str, Any]] = []
    word_start = text_start = 0
    for count in range(1, len(words) + 1):
        if count not in boundaries and count < len(words):
            if words[count - 1]["end"] - words[word_start]["start"] < 12:
                continue
            boundaries[count] = words[count]["text_start"]
        text_end = boundaries.get(count, len(text))
        segments.append({
            "start": words[word_start]["start"], "end": words[count - 1]["end"],
            "text": text[text_start:text_end], "text_start": text_start, "text_end": text_end,
            "word_start": word_start, "word_end": count, "region_index": region_index,
        })
        word_start, text_start = count, text_end
    if "".join(segment["text"] for segment in segments) != text:
        raise AlignmentOutputError("Segment source-span coverage mismatch")
    return segments


def _validate_subtitle_text(text: str) -> None:
    line_check = text.replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"\n[ \t]*\n", line_check) or "-->" in text:
        raise AlignmentInputError("Transcript contains blank cue lines or '-->' that cannot be preserved in subtitles; use verbose_json")


def output(result: AlignmentResult, response_format: str, timestamp: Callable[[float], str]) -> dict[str, Any] | str:
    if response_format == "verbose_json":
        return result.payload()
    if not isinstance(response_format, str) or response_format not in RESPONSE_FORMATS:
        raise AlignmentInputError("Alignment response_format must be verbose_json, srt, or vtt")
    if result.diagnostics["zero_duration_word_indices"]:
        raise AlignmentOutputError("Zero-duration alignment words cannot be exported as subtitles; use verbose_json to inspect diagnostics")
    lines = ["WEBVTT", ""] if response_format == "vtt" else []
    for index, segment in enumerate(result.segments, 1):
        if round(segment["end"] * 1000) <= round(segment["start"] * 1000):
            raise AlignmentOutputError("Subtitle cue collapses at millisecond resolution; use verbose_json to inspect timing")
        content = segment["text"]
        _validate_subtitle_text(content)
        content = html.escape(content, quote=False)
        # A trailing line ending is also the cue's terminator, not an extra blank cue.
        if response_format == "srt":
            lines.append(str(index))
        lines.append(f"{timestamp(segment['start'])} --> {timestamp(segment['end'])}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines) + "\n"
