from __future__ import annotations

import html
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import Cue
from .writers import write_ass, write_srt, write_subtitle_pair_exclusive


class SubtitleFormatError(ValueError):
    pass


def decode_subtitle_bytes(data: bytes) -> str:
    if not data:
        raise SubtitleFormatError("empty subtitle file")
    if len(data) > 20 * 1024 * 1024:
        raise SubtitleFormatError("subtitle file is unexpectedly large")
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    raise SubtitleFormatError("unsupported subtitle text encoding")


def detect_subtitle_format(text: str, url: str = "") -> str:
    head=text.lstrip()[:512].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or head.startswith("<head"):
        raise SubtitleFormatError("received HTML instead of a subtitle")
    if re.search(r"^\s*\[script info\]", text, re.I|re.M):
        return "ass"
    if head.startswith("webvtt"):
        return "vtt"
    if "-->" in text and re.search(r"^\s*\d+\s*$", text, re.M):
        return "srt"
    suffix=Path(url.split("?",1)[0]).suffix.lower()
    if suffix in {".srt",".vtt",".ass",".ssa"}:
        return suffix.lstrip(".").replace("ssa","ass")
    raise SubtitleFormatError("unknown subtitle format")


def _parse_clock(value: str) -> float:
    original=value
    value=value.strip().replace(",", ".")
    parts=value.split(":")
    if len(parts)==3:
        h,m,s=parts
    elif len(parts)==2:
        h="0";m,s=parts
    else:
        raise SubtitleFormatError(f"invalid timestamp: {original}")
    try:
        hours=int(h)
        minutes=int(m)
        seconds=float(s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubtitleFormatError(f"invalid timestamp: {original}") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise SubtitleFormatError(f"invalid timestamp: {original}")
    return hours*3600 + minutes*60 + seconds


def _clean_text(lines: list[str]) -> str:
    out=[]
    for line in lines:
        line=line.strip()
        if line:
            out.append(line)
    return "\n".join(out).strip()


def parse_srt(text: str) -> list[Cue]:
    blocks=re.split(r"\n\s*\n",text.replace("\r\n","\n").replace("\r","\n").strip())
    cues=[]
    timing=re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")
    for block in blocks:
        lines=block.splitlines()
        if not lines: continue
        idx=1 if lines[0].strip().isdigit() else 0
        if idx>=len(lines): continue
        m=timing.match(lines[idx])
        if not m: continue
        body=_clean_text(lines[idx+1:])
        if body:
            cues.append(Cue(_parse_clock(m.group(1)),_parse_clock(m.group(2)),body))
    if not cues:
        raise SubtitleFormatError("no SRT cues found")
    return cues


def _clean_vtt_text(lines: list[str]) -> str:
    """Convert WebVTT cue payload to plain text without losing line breaks.

    Remove markup before decoding entities: &lt;br&gt; is literal subtitle
    content, whereas an actual <br> cue tag denotes a line break. Keep the
    base text of ruby annotations without duplicating its pronunciation.
    """
    body = "\n".join(lines)
    body = re.sub(
        r"<(?:rt|rp)(?:\s[^>\n]*)?>.*?</(?:rt|rp)\s*>",
        "", body, flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<[^>\n]*>", "", body)
    return _clean_text(html.unescape(body).splitlines())


def parse_vtt(text: str) -> list[Cue]:
    lines=text.replace("\r\n","\n").replace("\r","\n").splitlines()
    cues=[]
    i=0
    timing=re.compile(r"^\s*((?:\d{1,2}:)?\d{2}:\d{2}\.\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}\.\d{1,3})")
    while i<len(lines):
        line=lines[i].strip()
        # WebVTT metadata/comment blocks are not cues, even when a NOTE
        # happens to contain a timestamp-shaped line.
        if re.fullmatch(r"NOTE(?:[ \t].*)?", line) or line in {"STYLE", "REGION"}:
            i+=1
            while i<len(lines) and lines[i].strip():
                i+=1
            continue
        m=timing.match(line)
        if not m and line and i+1<len(lines):
            # optional cue identifier
            m=timing.match(lines[i+1].strip())
            if m: i+=1
        if m:
            start,end=_parse_clock(m.group(1)),_parse_clock(m.group(2))
            i+=1;body=[]
            while i<len(lines) and lines[i].strip():
                body.append(lines[i])
                i+=1
            txt=_clean_vtt_text(body)
            if txt:cues.append(Cue(start,end,txt))
        i+=1
    if not cues:
        raise SubtitleFormatError("no VTT cues found")
    return cues

def _ass_visible_text(body: str) -> str:
    r"""Decode ASS display text without interpreting escaped literal braces.

    Unlike deleting every {...} span with a regex, this must distinguish
    real override blocks from FFmpeg's \{{} literal opening brace and
    libass's \{ / \} extension. The output always represents visible text.
    """
    out = []
    index = 0
    while index < len(body):
        if body.startswith(r"\{{}", index):
            out.append("{")
            index += 4
            continue
        if body.startswith(r"\{", index):
            out.append("{")
            index += 2
            continue
        if body.startswith(r"\}", index):
            out.append("}")
            index += 2
            continue
        char = body[index]
        if char == "{":
            end = body.find("}", index + 1)
            if end != -1:
                # Genuine ASS formatting/comment block, not visible glyphs.
                index = end + 1
                continue
        if char == "\\" and index + 1 < len(body):
            following = body[index + 1]
            if following == "\u2060":
                # FFmpeg word-joiner prevents literal slash+N/H glyphs from
                # unexpectedly becoming ASS linebreak/hard-space commands.
                out.append("\\")
                index += 2
                continue
            if following == "\\":
                # Backwards compatibility for legacy HSE ASS exports.
                out.append("\\")
                index += 2
                continue
            if following in ("N", "n"):
                out.append("\n")
                index += 2
                continue
            if following == "h":
                out.append(" ")
                index += 2
                continue
        out.append(char)
        index += 1
    return "".join(out)


def parse_ass(text: str) -> list[Cue]:
    lines=text.replace("\r\n","\n").replace("\r","\n").splitlines()
    in_events=False
    fields=None
    cues=[]
    for line in lines:
        stripped=line.strip()
        if stripped.lower()=="[events]":
            in_events=True;continue
        if stripped.startswith("[") and stripped.endswith("]") and stripped.lower()!="[events]":
            in_events=False
        if not in_events: continue
        if stripped.lower().startswith("format:"):
            fields=[x.strip().lower() for x in stripped.split(":",1)[1].split(",")]
            continue
        if not stripped.lower().startswith("dialogue:"):
            continue
        payload=stripped.split(":",1)[1].lstrip()
        if fields:
            parts=payload.split(",",len(fields)-1)
            if len(parts)<len(fields):continue
            row=dict(zip(fields,parts))
            start=row.get("start","");end=row.get("end","");body=row.get("text","")
        else:
            parts=payload.split(",",9)
            if len(parts)<10:continue
            start,end,body=parts[1],parts[2],parts[9]
        body=_ass_visible_text(body)
        body=_clean_text(body.splitlines())
        if body:
            cues.append(Cue(_parse_clock(start),_parse_clock(end),body))
    if not cues:
        raise SubtitleFormatError("no ASS dialogue cues found")
    return cues


def _validate_cues(cues: list[Cue]) -> None:
    for cue in cues:
        if (
            not math.isfinite(cue.start)
            or not math.isfinite(cue.end)
            or cue.start < 0
            or cue.end <= cue.start
        ):
            raise SubtitleFormatError("subtitle contains an invalid cue timeline")


def parse_subtitle_bytes(
    data: bytes,
    *,
    source_url: str = "",
) -> tuple[str, list[Cue]]:
    """Decode and parse a subtitle without creating output files."""
    text=decode_subtitle_bytes(data)
    fmt=detect_subtitle_format(text,source_url)
    if fmt=="srt":
        cues=parse_srt(text)
    elif fmt=="vtt":
        cues=parse_vtt(text)
    elif fmt=="ass":
        cues=parse_ass(text)
    else:
        raise SubtitleFormatError(fmt)
    _validate_cues(cues)
    return fmt,cues


def subtitle_cues_have_korean(cues: list[Cue]) -> bool:
    """Reject clearly non-Korean tracks using bounded Hangul density evidence."""
    sample = cues[:200]
    if not sample:
        return False
    # Normalize only the language-detection view. Preserve the original cue
    # text for output so this check cannot rewrite user-facing subtitles.
    text = unicodedata.normalize("NFC", "\n".join(cue.text for cue in sample))
    hangul_count = len(re.findall(r"[가-힣]", text))
    if hangul_count < 2:
        return False
    letter_count = sum(char.isalpha() for char in text)
    ratio = hangul_count / max(letter_count, 1)
    # Keep a ratio requirement even for longer files so a foreign-language
    # subtitle with one Korean title/note cannot suppress the hard-OCR path.
    return ratio >= 0.15


def subtitle_bytes_have_korean(
    data: bytes,
    *,
    source_url: str = "",
) -> bool:
    """Validate a soft subtitle's language without writing SRT/ASS outputs."""
    _,cues=parse_subtitle_bytes(data,source_url=source_url)
    return subtitle_cues_have_korean(cues)


@dataclass(slots=True)
class NormalizedSubtitle:
    source_format: str
    cue_count: int
    srt_path: Path
    ass_path: Path


def normalize_subtitle_bytes(
    data: bytes,
    out_base: str | Path,
    *,
    source_url: str = "",
    require_korean: bool = False,
    exclusive_outputs: bool = False,
) -> NormalizedSubtitle:
    fmt,cues=parse_subtitle_bytes(data,source_url=source_url)

    # Language validation must happen before either writer so a valid but
    # wrong-language soft track cannot leave files that later block a retry.
    if require_korean and not subtitle_cues_have_korean(cues):
        raise SubtitleFormatError("subtitle does not contain enough Korean text")

    base=Path(out_base);base.parent.mkdir(parents=True,exist_ok=True)
    if exclusive_outputs:
        srt,ass=write_subtitle_pair_exclusive(cues,base)
    else:
        srt=write_srt(cues,base.with_suffix(".srt"))
        ass=write_ass(cues,base.with_suffix(".ass"))
    return NormalizedSubtitle(fmt,len(cues),srt,ass)
