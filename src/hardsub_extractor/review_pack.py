"""Opt-in local-only original-frame review for uncertain extracted captions.

Never transcribes, edits, deletes, uploads, or transmits source media. The
generated JPEGs contain PRIVATE source imagery; keep the output owner-private.
"""
from __future__ import annotations

import argparse
from html import escape as html_escape
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

from .quality_audit import AuditOptions, audit_subtitles
from .runtime_paths import resolve_binary
from .subtitle_language import normalize_language
from .quality_audit_io import _read_subtitles, _probe_local_video_duration
from .local_ocr_quality_signals import REVIEW_CODES

MAX_CUES = 12
MAX_FRAME_BYTES = 2_500_000


def select_cues(cues, report, mode: str, limit: int, *,
                after_cue: int = 0,
                temporal_cue_numbers: tuple[int, ...] | None = None,
                split_cue_numbers: tuple[int, ...] | None = None,
                ) -> list[tuple[int, list[str]]]:
    """Keep original 1-based cue identities; warning labels are advisory."""
    if (mode not in {"flagged", "all", "temporal", "quality", "split"} or not 1 <= limit <= MAX_CUES
            or type(after_cue) is not int or not 0 <= after_cue <= len(cues)):
        raise ValueError("invalid_review_selection")
    warnings: dict[int, set[str]] = {}
    for code, entries in report["candidate"]["review_locations"].items():
        if not isinstance(entries, list):
            raise ValueError("invalid_warning_list")
        for entry in entries:
            cue = entry.get("cue")
            if type(cue) is not int or not 1 <= cue <= len(cues):
                raise ValueError("invalid_warning_cue_index")
            warnings.setdefault(cue, set()).add(code)
    if mode == "all":
        indices = range(after_cue + 1, min(len(cues), after_cue + limit) + 1)
    elif mode == "temporal":
        # Candidate identities are produced by the same local OCR worker and
        # pinned to its SRT in the GUI; they are suggestions, not deletions.
        if (not isinstance(temporal_cue_numbers, tuple)
            or any(type(n) is not int or not 1 <= n <= len(cues)
                   for n in temporal_cue_numbers)
            or len(set(temporal_cue_numbers)) != len(temporal_cue_numbers)):
            raise ValueError("invalid_or_stale_temporal_review_hints")
        indices = sorted(
            n for n in temporal_cue_numbers if n > after_cue)[:limit]
        for n in indices:
            warnings.setdefault(n,set()).add(
                "temporal_persistent_overlay_review")
    elif mode == "split":
        # Pair starts come only from this extraction's attested RAM receipt.
        # Never treat these numbers as user-supplied edit/deletion commands.
        if (not isinstance(split_cue_numbers, tuple)
                or len(split_cue_numbers) > 8):
            raise ValueError("invalid_split_review_hints")
        last_start = -1
        for start in split_cue_numbers:
            if (
                type(start) is not int or not 1 <= start < len(cues)
                or start <= last_start + 1
            ):
                raise ValueError("invalid_or_overlapping_split_pair")
            last_start = start
        indices = [
            cue for start in split_cue_numbers
            for cue in (start, start + 1) if cue > after_cue
        ][:limit]
        for number in indices:
            warnings.setdefault(number, set()).add("native_split_pair_review")
    elif mode == "quality":
        # Use precisely the same major-warning policy that provided the
        # post-export cue numbers. Re-evaluate after_cue for the next page;
        # do not mix unrelated script/language/long-gap advisory warnings.
        indices = sorted(
            i for i, codes in warnings.items()
            if i > after_cue and codes.intersection(REVIEW_CODES)
        )[:limit]
    else:
        indices = sorted(i for i in warnings if i > after_cue)[:limit]
    return [
        (index, sorted(
            warnings.get(index, set()).intersection(REVIEW_CODES)
            if mode == "quality" else warnings.get(index, set())
        ))
        for index in indices
    ]


def sampling_times(cue, duration: float) -> tuple[float, float]:
    if not (math.isfinite(cue.start) and math.isfinite(cue.end)
            and 0 <= cue.start < cue.end and cue.start < duration - 0.05):
        raise ValueError("invalid_review_cue_timeline")
    points = tuple(min(
        duration - 0.05, cue.start + (cue.end - cue.start) * fraction,
    ) for fraction in (0.20, 0.80))
    if not all(cue.start <= sec < duration for sec in points):
        raise ValueError("review_sample_outside_media")
    return points


def extract_frame(video: Path, seconds: float) -> bytes:
    """Only local file/pipe FFmpeg protocols, bounded output and time."""
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe",
        "-ss", f"{seconds:.3f}", "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-frames:v", "1", "-vf", "scale=w='min(iw,960)':h=-2",
        "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False, timeout=35,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("private_review_decode_failed") from exc
    data = result.stdout
    if result.returncode or not 100 <= len(data) <= MAX_FRAME_BYTES:
        raise ValueError("private_review_frame_unavailable")
    try:
        with Image.open(BytesIO(data)) as image:
            if (image.format != "JPEG" or not 1 <= image.width <= 960
                    or not 1 <= image.height <= 1440):
                raise ValueError("private_review_frame_bad_shape")
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("private_review_frame_invalid") from exc
    return data


def contact_sheet(images: list[tuple[int, str, bytes]]) -> bytes:
    """Make a small numbered visual overview without embedding OCR text."""
    width, row_height, header = 480, 294, 24
    count = (len(images) + 1) // 2
    if not 1 <= count <= MAX_CUES:
        raise ValueError("invalid_private_review_count")
    canvas = Image.new("RGB", (width * 2, row_height * count), "#202020")
    draw = ImageDraw.Draw(canvas)
    for item, (cue, stage, raw) in enumerate(images):
        x, y = item % 2 * width, item // 2 * row_height
        with Image.open(BytesIO(raw)) as image:
            image.load()
            image.thumbnail((width, row_height - header))
            canvas.paste(image.convert("RGB"), (x, y + header))
        draw.text((x + 6, y + 5), f"Cue {cue} / {stage}", fill="white")
    output = BytesIO()
    canvas.save(output, format="JPEG", quality=83, optimize=True)
    return output.getvalue()


def private_review_page(cues, selected) -> bytes:
    """Owner-private local report: original SRT words + independently decoded frames.

    This is the only additional text-bearing output. Existing index.json
    remains text-free; no web resources, scripts, uploads or SRT changes.
    """
    if not 1 <= len(selected) <= MAX_CUES:
        raise ValueError("private_review_selection_out_of_bounds")
    blocks = []
    for number, codes in selected:
        if type(number) is not int or not 1 <= number <= len(cues):
            raise ValueError("invalid_private_review_cue")
        cue = cues[number - 1]
        original = cue.text[:8192]
        body = html_escape(original, quote=True)
        warnings = html_escape(" · ".join(codes), quote=True)
        shortened = "<p>긴 OCR 문장은 일부만 표시 · 원본 SRT를 확인하세요.</p>" if len(cue.text) > len(original) else ""
        figures = "".join(
            f'<figure><img src="cue-{number:05d}-{stage}.jpg" alt="{label} 원본">'
            f'<figcaption>{label} 프레임</figcaption></figure>'
            for stage, label in (("early", "앞쪽"), ("late", "뒤쪽"))
        )
        blocks.append(
            f'<section><h2>자막 {number}번 · {cue.start:.3f}–{cue.end:.3f}초</h2>'
            f'<p class="warn">{warnings}</p><pre>{body}</pre>{shortened}'
            f'<div class="frames">{figures}</div></section>'
        )
    page = (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="'
        "default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; "
        "script-src 'none'; connect-src 'none'; object-src 'none'; "
        "base-uri 'none'; form-action 'none';"
        '"><title>HSE · 개인 원본 화면 검토</title><style>'
        'body{font:16px/1.6 system-ui,"Malgun Gothic",sans-serif;'
        'background:#161b23;color:#f1f4f8;max-width:1060px;margin:0 auto;padding:18px}'
        'h1{font-size:1.45rem}h2{font-size:1.1rem}p{color:#d7e1ee}'
        'section{background:#242c38;border:1px solid #56657b;'
        'border-radius:12px;margin:16px 0;padding:16px}'
        'pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;'
        'background:#121923;border-left:3px solid #87b3f9;padding:12px}'
        '.warn{font-size:.9rem}.frames{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:12px}'
        'figure{margin:0}img{max-width:100%;height:auto}figcaption{color:#d7e1ee}'
        '</style></head><body><h1>HSE · 자막과 원본 프레임 비교</h1>'
        '<p>이 파일은 내 컴퓨터에만 저장됩니다. OCR 글자가 정확한 대사임을 '
        '보증하지 않습니다. 교정은 원본 SRT가 아닌 별도 사본에서 진행하세요.</p>'
        + "".join(blocks) + '</body></html>'
    )
    data = page.encode("utf-8")
    if not 0 < len(data) <= 128 * 1024:
        raise ValueError("private_review_page_over_budget")
    return data


def create_pack(video: Path, subtitle: Path, output: Path, *,
                mode: str = "flagged", max_cues: int = 8,
                language: str = "ko", after_cue: int = 0,
                check_cancelled=None,
                temporal_cue_numbers: tuple[int, ...] | None = None,
                split_cue_numbers: tuple[int, ...] | None = None) -> dict:
    """Create a fresh private folder only after decoding all selected frames.

    Failure never truncates existing files, and cleanup touches only files
    created inside the directory exclusively owned by this invocation.
    """
    video, subtitle, output = Path(video), Path(subtitle), Path(output)
    if check_cancelled is not None:
        check_cancelled()
    if output.exists() or output.is_symlink():
        raise ValueError("review_output_exists")
    cues = _read_subtitles(subtitle)
    duration = _probe_local_video_duration(video)
    report = audit_subtitles(cues, options=AuditOptions(
        requested_language=normalize_language(language),
        video_duration_seconds=duration,
        max_review_locations=max_cues,
        review_locations_after_cue=after_cue,
    ))
    selected = select_cues(
        cues, report, mode, max_cues, after_cue=after_cue,
        temporal_cue_numbers=temporal_cue_numbers,
        split_cue_numbers=split_cue_numbers)
    if not selected:
        return {"status": "NO_REVIEW_LOCATIONS", "cue_count": 0,
                "frame_count": 0, "last_cue": after_cue}
    images: list[tuple[int, str, bytes]] = []
    index: list[dict] = []
    for number, codes in selected:
        if check_cancelled is not None:
            check_cancelled()
        cue = cues[number - 1]
        frames = []
        for stage, second in zip(("early", "late"), sampling_times(cue, duration)):
            if check_cancelled is not None:
                check_cancelled()
            raw = extract_frame(video, second)
            name = f"cue-{number:05d}-{stage}.jpg"
            images.append((number, stage, raw))
            frames.append({
                "image": name, "second": round(second, 3),
                "sha256": sha256(raw).hexdigest(),
            })
        index.append({
            "cue": number, "start": round(cue.start, 3),
            "end": round(cue.end, 3), "warning_codes": codes,
            "frames": frames,
        })
    if check_cancelled is not None:
        check_cancelled()
    overview = contact_sheet(images)
    private_page = private_review_page(cues, selected)
    manifest = (json.dumps({
        "schema": 1, "review_only": True, "private_local_images": True,
        "no_caption_text_embedded": True, "mode": mode,
        "after_cue": after_cue,
        "original_cue_count": len(cues), "review_cues": index,
    }, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    created: list[Path] = []
    try:
        for cue, stage, data in images:
            if check_cancelled is not None:
                check_cancelled()
            path = output / f"cue-{cue:05d}-{stage}.jpg"
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
        for name, data in (("contact-sheet.jpg", overview), ("index.json", manifest),
                           ("review.html", private_page)):
            if check_cancelled is not None:
                check_cancelled()
            path = output / name
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except Exception:
        for path in reversed(created):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        output.rmdir()
        raise
    return {"status": "CREATED", "cue_count": len(selected),
            "frame_count": len(images), "last_cue": selected[-1][0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicit offline original-frame audit; outputs private local images",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("flagged", "all"), default="flagged")
    parser.add_argument("--max-cues", type=int, default=8)
    parser.add_argument("--after-cue", type=int, default=0,
                        help="Resume strictly after this original cue number")
    parser.add_argument("--language", default="ko")
    args = parser.parse_args(argv)
    try:
        result = create_pack(
            args.video, args.candidate, args.output,
            mode=args.mode, max_cues=args.max_cues, language=args.language,
            after_cue=args.after_cue,
        )
    except (OSError, ValueError) as exc:
        # Never print paths, video frames, subtitles or raw FFmpeg errors.
        print(f"HSE_REVIEW_PACK_FAILED={type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        "HSE_REVIEW_PACK=" + result["status"]
        + f";CUES={result['cue_count']};FRAMES={result['frame_count']}"
        + f";LAST_CUE={result['last_cue']}"
        + ";SOURCE_WRITE=NONE;NETWORK=NONE;PRIVATE_IMAGES=LOCAL"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
