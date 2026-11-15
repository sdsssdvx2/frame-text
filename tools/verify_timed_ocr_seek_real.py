"""Real-codec timing acceptance for HSE's optional sparse timed OCR.

No OCR model, external captions, website credentials or user files. Generate a
45-second moving test pattern using the SHA-pinned FFmpeg on the isolated CI
runner, compare all cue-region pixel hashes and frame timestamps against a
continuous full decode, then compare complete OCR export bytes using a fake
recognizer. No reference text is supplied to the recognizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

from hardsub_extractor.frame_source import (
    FFmpegPTSFrameSource, NormalizedROI,
)
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor import timed_ocr


@dataclass(frozen=True)
class Interval:
    start: float
    end: float


class FakeRecognizer:
    def __init__(self):
        self.samples = []

    def recognize(self, frame):
        digest = sha256(frame.image.tobytes()).hexdigest()
        self.samples.append((frame.pts, digest))
        return OCRObservation("OCR_ONLY", 0.99, frame.pts)


def _fingerprint(frame):
    return frame.pts, sha256(frame.image.tobytes()).hexdigest()


def _in_cue(frame, cues):
    return any(s.start <= frame.pts < s.end for s in cues)


def _collect(iterator, cues):
    selected = []
    frames = 0
    try:
        for frame in iterator:
            frames += 1
            if _in_cue(frame, cues):
                selected.append(_fingerprint(frame))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return selected, frames


def _same_samples(before, after):
    if len(before) != len(after):
        return False
    return all(
        abs(left[0] - right[0]) <= 0.00001 and left[1] == right[1]
        for left, right in zip(before, after)
    )


def run():
    cues = (Interval(1.333, 2.208), Interval(41.333, 42.208))
    spans = timed_ocr._validate_intervals(cues)
    roi = NormalizedROI(0.0, 0.0, 1.0, 1.0)
    with tempfile.TemporaryDirectory(prefix="hse-timed-seek-") as root:
        folder = Path(root)
        video = folder / "moving-synthetic.mkv"
        command = [
            resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
            "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc2=size=160x90:rate=24:duration=45",
            "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p",
            "-y", str(video),
        ]
        subprocess.run(command, check=True, timeout=90,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not video.is_file() or video.stat().st_size == 0:
            raise AssertionError("synthetic encoded source missing")
        before_sha = sha256(video.read_bytes()).hexdigest()

        for fps in (8.0, 10.0, 7.7):
            # Real production FFmpegPTSFrameSource decoders, not a mock.
            baseline, full_count = _collect(iter(FFmpegPTSFrameSource(
                str(video), roi=roi, sample_fps=fps,
            )), cues)
            sparse, sparse_count = _collect(iter(timed_ocr._interval_frames(
                str(video), spans, roi=roi, sample_fps=fps,
                check_cancelled=None,
            )), cues)
            if not baseline or not _same_samples(baseline, sparse):
                raise AssertionError(
                    f"real FFmpeg source PTS or pixel mismatch for fps={fps}"
                )
            if not (0 < sparse_count < full_count / 3):
                raise AssertionError("sparse decoder did not reduce frames")

            original = FakeRecognizer()
            seeked = FakeRecognizer()
            baseline_base = folder / f"baseline-{fps}"
            seeked_base = folder / f"sparse-{fps}"
            # The reference is HSE's *existing full decoder*, not a subtitle
            # oracle or synthetic OCR transcript read by a model.
            with mock.patch.object(
                timed_ocr, "_decoder_windows",
                return_value=((0.0, 44.5),),
            ):
                before = timed_ocr.extract_timed_ocr(
                    str(video), cues, original, baseline_base,
                    sample_fps=fps, roi=roi,
                )
            after = timed_ocr.extract_timed_ocr(
                str(video), cues, seeked, seeked_base,
                sample_fps=fps, roi=roi,
            )
            if not _same_samples(original.samples, seeked.samples):
                raise AssertionError("timed OCR selected different video pixels")
            if before.detected_intervals != after.detected_intervals:
                raise AssertionError("timed OCR cue count changed")
            if (before.recognized_intervals, before.missed_intervals,
                before.ocr_calls) != (after.recognized_intervals,
                                      after.missed_intervals, after.ocr_calls):
                raise AssertionError("timed OCR coverage/call count changed")
            if before.srt_path.read_bytes() != after.srt_path.read_bytes():
                raise AssertionError("timed OCR SRT output changed")
            if before.ass_path.read_bytes() != after.ass_path.read_bytes():
                raise AssertionError("timed OCR ASS output changed")
            _, parsed = parse_subtitle_bytes(
                after.srt_path.read_bytes(), source_url="verified.srt",
            )
            if len(parsed) != 2 or any(
                abs(c.start - x.start) > 0.001
                or abs(c.end - x.end) > 0.001
                for c, x in zip(parsed, cues)
            ):
                raise AssertionError("subtitles lost original source timeline")
            print(f"FPS={fps};FULL_FRAMES={full_count};SPARSE_FRAMES="
                  f"{sparse_count};CUE_PIXEL_AND_TEXT=PASS")

        if sha256(video.read_bytes()).hexdigest() != before_sha:
            raise AssertionError("source video was modified")
        print("RESULT=REAL_FFMPEG_SEEK_PTS_PIXEL_EXPORT_MATCH;SOURCE_UNCHANGED=YES")


if __name__ == "__main__":
    run()
