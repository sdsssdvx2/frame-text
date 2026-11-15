"""Offline, read-only A/B scorer for separately obtained subtitle OCR exports.

No external engines are installed or invoked. This grades SRT exports against an
independently annotated reference on one identical source video and paired
blank-video controls. An input SHA attestation is NOT execution provenance.
Research only: this result cannot authorize an HSE production release.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata

MAX_MANIFEST_BYTES = 96 * 1024
MAX_SRT_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 5 * 1024 * 1024 * 1024
MAX_CUES = 3000
MAX_ENGINES = 12
LANGUAGES = {"ko", "en", "ja", "zh"}
TIME_RE = re.compile(r"^(\d{1,4}):(\d{2}):(\d{2})[,.](\d{3})$")
ARROW_RE = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)\s*$")
ENGINE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".ts"}


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _reject_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON field")
        obj[key] = value
    return obj


def safe_path(root: Path, rel: str, endings: set[str]) -> Path:
    if not isinstance(rel, str) or not rel or len(rel) > 220:
        raise ValueError("invalid relative file")
    if "\\" in rel or ":" in rel or "\x00" in rel or "//" in rel:
        raise ValueError("unsupported file path")
    p = PurePosixPath(rel)
    if p.is_absolute() or any(x in (".", "..") for x in rel.split("/")):
        raise ValueError("unsafe file traversal")
    if p.suffix.lower() not in endings:
        raise ValueError("unexpected input file extension")
    output = root.joinpath(*p.parts)
    if output.is_symlink() or any(
        parent.is_symlink() for parent in output.parents
        if parent != root and parent != root.parent
    ):
        raise ValueError("symlinked input rejected")
    real = output.resolve(strict=True)
    if not real.is_relative_to(root) or not real.is_file():
        raise ValueError("input outside study directory")
    return real


def limited_read(path: Path, ceiling: int) -> bytes:
    if path.stat().st_size > ceiling:
        raise ValueError("input file exceeds size limit")
    with path.open("rb") as f:
        data = f.read(ceiling + 1)
    if len(data) > ceiling:
        raise ValueError("input grew beyond limit")
    return data


def digest_video(path: Path) -> str:
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("video file missing or exceeds 5-GiB bound")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clock(value):
    m = TIME_RE.fullmatch(value)
    if m is None:
        raise ValueError("invalid SRT timestamp")
    hh, mm, ss, ms = map(int, m.groups())
    if mm >= 60 or ss >= 60:
        raise ValueError("invalid SRT clock range")
    return hh * 3600 + mm * 60 + ss + ms / 1000


def parse_strict_srt(data: bytes, *, allow_empty: bool = False) -> list[Cue]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("SRT must be UTF-8") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        if allow_empty:
            return []
        raise ValueError("reference/positive SRT must contain cues")
    cues = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = block.split("\n")
        if len(lines) < 2:
            raise ValueError("truncated SRT cue")
        if lines[0].isdigit():
            lines.pop(0)
        if len(lines) < 2:
            raise ValueError("cue without timing or text")
        times = ARROW_RE.fullmatch(lines[0])
        if times is None:
            raise ValueError("non-SRT block detected; silently dropping cues forbidden")
        start, end = _clock(times[1]), _clock(times[2])
        if end <= start or start < 0:
            raise ValueError("non-positive SRT duration")
        body = "\n".join(line.strip() for line in lines[1:]).strip()
        if not body or "\x00" in body or len(body) > 600:
            raise ValueError("missing or invalid SRT cue body")
        if cues and start < cues[-1].start:
            raise ValueError("subtitle cue timestamps not ordered")
        cues.append(Cue(start, end, unicodedata.normalize("NFC", body)))
        if len(cues) > MAX_CUES:
            raise ValueError("too many subtitle cues")
    return cues


def edit_distance(a: str, b: str) -> int:
    # Per-cue bounded dynamic program; no user text leaves the scorer.
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def align_cues(truth: list[Cue], observed: list[Cue]) -> list[tuple[int, int]]:
    """Order-preserving 1:1 optimal overlap match, NEVER text-oracle aligned.

    Only sufficiently overlapping times match. Detection score is independent
    of the reference text, so transcription cannot rescue wrong boundaries.
    """
    n, m = len(truth), len(observed)
    if not n or not m:
        return []
    weight = (min(n, m) + 1) * 1001
    prev = [0] * (m + 1)
    trace = [bytearray(m + 1) for _ in range(n + 1)]
    for i, ref in enumerate(truth, 1):
        row = [0] * (m + 1)
        for j, hyp in enumerate(observed, 1):
            up, left = prev[j], row[j - 1]
            if up >= left:
                best, choice = up, 1
            else:
                best, choice = left, 2
            overlap = max(0.0, min(ref.end, hyp.end) - max(ref.start, hyp.start))
            union = max(ref.end, hyp.end) - min(ref.start, hyp.start)
            if overlap > 0 and union > 0 and overlap / union >= 0.10:
                candidate = prev[j - 1] + weight + int(overlap / union * 1000)
                if candidate > best:
                    best, choice = candidate, 3
            row[j] = best
            trace[i][j] = choice
        prev = row
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        direction = trace[i][j]
        if direction == 3:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _flatten(text: str) -> str:
    return "".join(text.splitlines())


def evaluate(truth: list[Cue], observed: list[Cue], blank: list[Cue]) -> dict:
    matches = align_cues(truth, observed)
    ref_matched = {i for i, _ in matches}
    hyp_matched = {j for _, j in matches}
    total = sum(len(_flatten(r.text)) for r in truth)
    changes = sum(
        len(_flatten(truth[i].text))
        for i in range(len(truth)) if i not in ref_matched
    )
    changes += sum(
        len(_flatten(observed[j].text))
        for j in range(len(observed)) if j not in hyp_matched
    )
    exact_text = exact_wrap = tight_250 = loose_650 = 0
    errs = []
    for i, j in matches:
        a, b = truth[i], observed[j]
        same = _flatten(a.text) == _flatten(b.text)
        wrap = same and a.text == b.text
        errors = (abs(a.start - b.start), abs(a.end - b.end))
        errs.extend(errors)
        exact_text += same
        exact_wrap += wrap
        tight_250 += wrap and max(errors) <= 0.25 + 1e-8
        loose_650 += wrap and max(errors) <= 0.65 + 1e-8
        changes += edit_distance(_flatten(a.text), _flatten(b.text))
    return {
        "reference_cues": len(truth),
        "extracted_cues": len(observed),
        "matched_cues": len(matches),
        "missed_reference_cues": len(truth) - len(matches),
        "extra_extracted_cues": len(observed) - len(matches),
        "exact_visible_text_ignoring_wrap": exact_text,
        "exact_visible_text_and_wrap": exact_wrap,
        "strict_both_edges_250ms_and_text_wrap": tight_250,
        "both_edges_650ms_and_text_wrap": loose_650,
        "aggregate_character_error_rate": round(changes / max(1, total), 5),
        "mean_absolute_edge_error_seconds": round(
            sum(errs) / len(errs), 5,
        ) if errs else None,
        "blank_video_false_cues": len(blank),
        "no_caption_control_pass": len(blank) == 0,
        "all_reference_cues_exact_both_edges_250ms": bool(
            truth and not blank and not changes
            and not (len(truth) - len(matches))
            and not (len(observed) - len(matches))
            and tight_250 == len(truth)
        ),
    }


def grade_directory(directory: Path) -> dict:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("study directory missing")
    manifest = safe_path(root, "study.json", {".json"})
    study = json.loads(
        limited_read(manifest, MAX_MANIFEST_BYTES),
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    if not isinstance(study, dict) or study.get("schema") != 1:
        raise ValueError("unsupported comparison manifest")
    if (
        study.get("language") not in LANGUAGES
        or study.get("reference_scope") not in {"synthetic", "human_verified_real"}
    ):
        raise ValueError("unknown language or reference provenance")
    entries = study.get("engines")
    if not isinstance(entries, list) or not 2 <= len(entries) <= MAX_ENGINES:
        raise ValueError("at least two competing engines required")
    pos = safe_path(root, study.get("source_video"), VIDEO_SUFFIXES)
    neg = safe_path(root, study.get("blank_video"), VIDEO_SUFFIXES)
    if pos == neg or study.get("source_sha256") == study.get("blank_sha256"):
        raise ValueError("caption and no-caption source are not independent")
    actual = digest_video(pos)
    blank_actual = digest_video(neg)
    if (
        not SHA_RE.fullmatch(str(study.get("source_sha256")))
        or not SHA_RE.fullmatch(str(study.get("blank_sha256")))
        or actual != study["source_sha256"]
        or blank_actual != study["blank_sha256"]
    ):
        raise ValueError("source or blank MP4 SHA-256 mismatch")
    reference_path = safe_path(root, study.get("reference_srt"), {".srt"})
    reference = parse_strict_srt(limited_read(reference_path, MAX_SRT_BYTES))
    outputs = {}
    seen = set()
    occupied = {reference_path}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed engine record")
        eid = entry.get("id")
        if (
            not isinstance(eid, str)
            or not ENGINE_RE.fullmatch(eid)
            or eid in seen
        ):
            raise ValueError("invalid or duplicate engine ID")
        seen.add(eid)
        if (
            entry.get("input_video_sha256") != actual
            or entry.get("blank_video_sha256") != blank_actual
        ):
            raise ValueError("engine did not attest exact video input digest")
        output_path = safe_path(root, entry.get("output_srt"), {".srt"})
        blank_path = safe_path(root, entry.get("blank_srt"), {".srt"})
        if (
            output_path in occupied
            or blank_path in occupied
            or output_path == blank_path
        ):
            raise ValueError("candidate SRT paths must be independently named")
        occupied.update((output_path, blank_path))
        output = parse_strict_srt(
            limited_read(output_path, MAX_SRT_BYTES), allow_empty=True,
        )
        blank = parse_strict_srt(
            limited_read(blank_path, MAX_SRT_BYTES), allow_empty=True,
        )
        outputs[eid] = evaluate(reference, output, blank)
    return {
        "schema": 1,
        "reference_scope": study["reference_scope"],
        "language": study["language"],
        "source_file_sha_verified": True,
        "blank_source_file_sha_verified": True,
        "engine_reported_source_digests_match": True,
        "source_used_by_engine_independently_attested": False,
        "third_party_engines_actually_executed_by_this_tool": False,
        "no_network_no_install_no_media_write": True,
        "real_anime_accuracy_demonstrated": False,
        "release_gate_passed": False,
        "comparisons": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = grade_directory(args.study_dir)
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
        # Never display filenames, private text, raw source SHA, exception strings.
        print("HSE_UPSTREAM_AB_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }))
        return 3
    print("HSE_UPSTREAM_AB_SUMMARY " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
