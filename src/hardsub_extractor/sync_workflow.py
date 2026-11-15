from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .models import Cue
from .subtitle_formats import parse_srt
from .visual_sync import (
    PiecewiseSyncResult,
    apply_piecewise_offsets,
    estimate_piecewise_offsets,
)
from .writers import write_ass, write_srt, write_subtitle_pair_exclusive


@dataclass(slots=True)
class SyncedSubtitleResult:
    cues: list[Cue]
    srt_path: Path
    ass_path: Path
    report_path: Path
    sync: PiecewiseSyncResult


def sync_srt_to_target(
    source_srt: str | Path,
    source_video: str,
    target_video: str,
    out_base: str | Path,
    *,
    source_extra_input_args: list[str] | None = None,
    target_extra_input_args: list[str] | None = None,
    sample_fps: float = 4.0,
    window_seconds: float = 8.0,
    anchor_spacing: float = 4.0,
    max_abs_offset: float = 20.0,
    exclusive_outputs: bool = False,
) -> SyncedSubtitleResult:
    source_srt = Path(source_srt)
    cues = parse_srt(source_srt.read_text(encoding="utf-8"))
    sync = estimate_piecewise_offsets(
        source_video,
        target_video,
        sample_fps=sample_fps,
        window_seconds=window_seconds,
        anchor_spacing=anchor_spacing,
        max_abs_offset=max_abs_offset,
        source_extra_input_args=source_extra_input_args,
        target_extra_input_args=target_extra_input_args,
    )
    mapped = apply_piecewise_offsets(cues, sync.segments)

    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    report = base.with_suffix(".sync.json")
    if exclusive_outputs and (report.exists() or report.is_symlink()):
        raise FileExistsError("기존 싱크 보고서가 있어 덮어쓸 수 없습니다.")
    if exclusive_outputs:
        srt, ass = write_subtitle_pair_exclusive(mapped, base)
    else:
        srt = write_srt(mapped, base.with_suffix(".srt"))
        ass = write_ass(mapped, base.with_suffix(".ass"))

    report_json = json.dumps(
            {
                "schema": 1,
                "source_subtitle": str(source_srt),
                "source_video": source_video,
                "target_video": target_video,
                "sample_fps": sync.sample_step and (1.0 / sync.sample_step),
                "segments": [
                    {
                        "source_start": x.source_start,
                        "source_end": x.source_end,
                        "offset": x.offset,
                        "anchor_count": x.anchor_count,
                        "mean_score": x.mean_score,
                    }
                    for x in sync.segments
                ],
                "anchors": [
                    {
                        "source_center": x.source_center,
                        "offset": x.offset,
                        "score": x.score,
                        "matched_frames": x.matched_frames,
                        "confidence_margin": x.confidence_margin,
                    }
                    for x in sync.anchors
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    if exclusive_outputs:
        # Protect the final sync report too. A late collision or write failure
        # rolls back only this job's *new* target subtitle pair; the original
        # unsynchronized source pair is preserved for independent re-use.
        report_created = False
        try:
            with report.open("x", encoding="utf-8") as output:
                report_created = True
                output.write(report_json)
        except BaseException:
            if report_created:
                try:
                    report.unlink(missing_ok=True)
                except OSError:
                    pass
            for generated in (srt, ass):
                try:
                    generated.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    else:
        report.write_text(report_json, encoding="utf-8")

    return SyncedSubtitleResult(mapped, srt, ass, report, sync)
