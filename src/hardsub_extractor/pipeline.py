from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
import math
from .models import Cue
from .cue_builder import merge_cues
from .refined_recognition import recognize_segments
from .detector import DetectorConfig,segment_frames,iter_segment_frames
from .frame_source import FFmpegPTSFrameSource,NormalizedROI
from .ocr.base import OCREngine
from .extraction_progress import ExtractionMonitor
from .writers import write_ass,write_srt,write_subtitle_pair_exclusive

@dataclass(slots=True)
class ExtractionReport:
    segment_count:int
    cue_count:int
    srt_path:Path
    ass_path:Path

def extract(
    source: str,
    ocr: OCREngine,
    out_base: str | Path,
    *,
    roi: NormalizedROI | None = None,
    detector: DetectorConfig | None = None,
    diagnostics=None,
    monitor: ExtractionMonitor | None = None,
    sample_fps: float | None = None,
    streaming: bool = False,
    timing_refiner=None,
    output_time_offset_ms: int = 0,
    focus_bottom_two_lines: bool = False,
) -> ExtractionReport:
    # Explicit post-recognition timeline offset for a partial tab recording.
    # No time shift is applied to decoded frames, segment detection, text
    # recognition or optical timing refinement. Legacy callers default to 0.
    if (type(output_time_offset_ms) is not int
            or not 0 <= output_time_offset_ms <= 99 * 3600 * 1000):
        raise ValueError("invalid_subtitle_time_offset")
    if type(focus_bottom_two_lines) is not bool:
        raise ValueError("visual OCR focus must be an explicit boolean")
    if monitor is not None and monitor.exclusive_outputs:
        # Fail before FFmpeg decoding; retain final exclusive publication.
        from .writers import ensure_subtitle_pair_available
        ensure_subtitle_pair_available(out_base)
    if focus_bottom_two_lines:
        from .ocr.line_focus import BottomTwoLineEngine
        ocr = BottomTwoLineEngine(ocr)
    # Existing workflows keep their original FFmpeg time sampling. The web
    # OCR path may explicitly limit frame samples to bound CPU and memory.
    frame_args = {"sample_fps": sample_fps} if sample_fps is not None else {}
    # Opt-in GUI cancel must interrupt a decoder blocked inside stdout.read,
    # not merely wait until a new frame arrives. Existing frame-source
    # supervision already reaps this exact child process on cancellation.
    # CLI/unmonitored pipelines retain their original no-watchdog behavior.
    if monitor is not None and callable(monitor.is_cancelled):
        frame_args["cancel_requested"] = monitor.is_cancelled
        # Only bound FFprobe for an existing user-owned local media file.
        # A remote/browser source may have a deliberately longer handshake;
        # its previously established probing behavior is left unchanged.
        try:
            local_media = Path(source).is_file()
        except (OSError, ValueError, TypeError):
            local_media = False
        if local_media:
            frame_args["probe_timeout_seconds"] = 60.0
    frames = FFmpegPTSFrameSource(source, roi=roi, **frame_args)
    if monitor is not None:
        # Streaming alternates frame capture and recognition; reporting both
        # stages separately would cause GUI flicker and misleading totals.
        frames = monitor.frames(frames, report_progress=not streaming)
    if streaming:
        # Work on one visual segment at a time. Never retain all decoded image
        # samples in memory before OCR. Always close the decoder even if
        # recognition, cancellation, export or model loading fails.
        segment_count = 0
        with ExitStack() as cleanup:
            frame_stream = iter(frames)
            # Some test and plugin iterators have no close() method. The real
            # FFmpeg generator does, and it MUST be closed even on OCR failure.
            frame_close = getattr(frame_stream, "close", None)
            if callable(frame_close):
                cleanup.callback(frame_close)
            stream = iter_segment_frames(frame_stream, detector)
            cleanup.callback(stream.close)

            def counted_segments():
                nonlocal segment_count
                for item in stream:
                    segment_count += 1
                    yield item

            if monitor is not None:
                # Number of OCR observations is unknown until all frames
                # have been decoded: progress has an open-ended total.
                ocr = monitor.ocr(ocr, total=None)
            cues = merge_cues(recognize_segments(
                counted_segments(), ocr, diagnostics=diagnostics,
                release_frames=True,
            ))
    else:
        segments = segment_frames(frames, detector)
        segment_count = len(segments)
        if monitor is not None:
            ocr = monitor.ocr(ocr, total=sum(len(segment.samples) for segment in segments))
        cues = merge_cues(recognize_segments(segments, ocr, diagnostics=diagnostics))
    if timing_refiner is not None and cues:
        # Strict opt-in contract. Never change original text, ordering,
        # confidence or cue count. Check BEFORE publishing SRT/ASS.
        refined = timing_refiner(cues, source, roi)
        if not isinstance(refined, list) or len(refined) != len(cues):
            raise ValueError("timing_refiner_changed_cue_count")
        previous_end = -1.0
        for before, after in zip(cues, refined):
            if (not isinstance(after, Cue)
                    or before.text != after.text
                    or before.confidence != after.confidence
                    or not all(math.isfinite(v) for v in (after.start, after.end))
                    or not 0 <= after.start < after.end
                    or after.start < previous_end - .005
                    or abs(after.start - before.start) > 2.20
                    or abs(after.end - before.end) > 2.20):
                raise ValueError("timing_refiner_broke_subtitle_invariants")
            previous_end = after.end
        cues = refined
    if not cues:
        raise RuntimeError("인식된 자막이 없습니다. 선택한 OCR 언어로 인식할 하드자막이 영상에 있는지 확인하세요.")
    if output_time_offset_ms:
        # Use new cue objects: do not mutate segment/recognition evidence.
        # Both canonical writers receive precisely the same shifted cues.
        delta = output_time_offset_ms / 1000.0
        cues = [
            Cue(c.start + delta, c.end + delta, c.text, c.confidence)
            for c in cues
        ]
    if monitor is not None:
        monitor.report("exporting", len(cues), len(cues))
    base=Path(out_base);base.parent.mkdir(parents=True,exist_ok=True)
    if monitor is not None and monitor.exclusive_outputs:
        srt,ass=write_subtitle_pair_exclusive(cues,base)
    else:
        srt=write_srt(cues,base.with_suffix(".srt"));ass=write_ass(cues,base.with_suffix(".ass"))
    if monitor is not None:
        monitor.finish(len(cues))
    return ExtractionReport(segment_count,len(cues),srt,ass)