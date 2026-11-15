"""Offline one-frame read-only OCR preflight; no model/network/browser."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor import local_ocr_preview as preview


class _ModifyOnExit:
    """Simulate a user replacing a clip exactly as its reader closes."""
    def __init__(self, stream, video):
        from contextlib import closing
        self.inner = closing(stream)
        self.video = video

    def __enter__(self):
        return self.inner.__enter__()

    def __exit__(self, *args):
        result = self.inner.__exit__(*args)
        self.video.write_bytes(b"the owner changed the synthetic source bytes")
        return result


class SampleOCRPreviewTests(unittest.TestCase):
    def test_all_region_presets_match_original_frame_decoder(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "movie.mp4"
            movie.write_bytes(b"local original unchanged")
            seen = []
            def stream(*args, **kwargs):
                seen.append(kwargs["roi"])
                yield Frame(0.0, np.zeros((12, 20, 3), dtype=np.uint8))
            class OCR:
                def recognize(self, frame):
                    return OCRObservation("대사", .92, frame.pts)
            with (mock.patch.object(preview, "probe_video",
                       return_value=mock.Mock(width=1920, height=1080, duration=10.0)),
                  mock.patch.object(preview, "FFmpegPTSFrameSource", side_effect=stream)):
                for key, expected in (
                    ("bottom40", (.04, .58, .92, .40)),
                    ("bottom25", (.04, .73, .92, .25)),
                    ("lower65", (.04, .33, .92, .65)),
                    ("lower80", (.04, .18, .92, .80)),
                ):
                    with self.subTest(key=key):
                        output = preview.inspect_local_ocr_preview(
                            movie, 0, lambda: OCR(), region_mode=key,
                        )
                        self.assertEqual(output.region_mode, key)
                        self.assertEqual((seen[-1].x, seen[-1].y, seen[-1].w, seen[-1].h), expected)
            self.assertEqual(movie.read_bytes(), b"local original unchanged")
            self.assertEqual(list(Path(td).iterdir()), [movie])

    def test_invalid_roi_never_starts_video_probe_or_model(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "movie.mp4"
            movie.write_bytes(b"unchanged")
            with mock.patch.object(preview, "probe_video") as probe:
                for wrong in ("", "bottom99", "https://example.org", None, True):
                    with self.subTest(region=wrong), self.assertRaises(ValueError):
                        preview.inspect_local_ocr_preview(
                            movie, 0, lambda: self.fail("model invoked"),
                            region_mode=wrong,
                        )
                probe.assert_not_called()

    def test_clock_grammar_and_bounds(self):
        for text, expected in (
            ("0", 0), ("15.25", 15.25), ("01:05", 65.0),
            ("01:02:03.5", 3723.5), ("06:00:00", 21600.0),
        ):
            with self.subTest(text=text):
                self.assertEqual(preview.parse_preview_second(text), expected)
        for text in (
            "", "12:60", "00:01:61", "1:2:3:4", "1.2:03",
            "-1", "nan", "inf", "06:00:00.1", "1,5", "http://a",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                preview.parse_preview_second(text)


    def test_comparison_reuses_one_video_frame_two_independent_ocr_passes(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "video.mp4"
            movie.write_bytes(b"original private media")
            decoded = []
            recognized = []
            def stream(*args, **kwargs):
                decoded.append(kwargs)
                yield Frame(0.0, np.full((20, 24, 3), 74, dtype=np.uint8))
            class Reading:
                def __init__(self, name):
                    self.name = name
                def recognize(self, frame):
                    recognized.append((self.name, frame.pts, frame.image.copy()))
                    frame.image[:] = 250
                    return OCRObservation(self.name, .94, frame.pts)
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=320, height=240, duration=15)),
                mock.patch.object(preview, "FFmpegPTSFrameSource", side_effect=stream),
            ):
                result = preview.inspect_local_ocr_preview(
                    movie, 0.0, lambda: Reading("ALL"),
                    preview_mode="compare-bright",
                    compare_make_ocr=lambda: Reading("BRIGHT"),
                )
            self.assertEqual(len(decoded), 1)
            self.assertEqual([x[0] for x in recognized], ["ALL", "BRIGHT"])
            self.assertTrue(np.array_equal(recognized[0][2], recognized[1][2]))
            self.assertEqual(result.text, "ALL")
            self.assertEqual(result.comparison_text, "BRIGHT")
            self.assertAlmostEqual(result.comparison_confidence, .94)
            self.assertTrue(np.all(result.image == 74))
            self.assertEqual(movie.read_bytes(), b"original private media")
            self.assertEqual(list(Path(td).iterdir()), [movie])

    def test_comparison_without_explicit_factory_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "movie.mp4"
            movie.write_bytes(b"untouched")
            with mock.patch.object(preview, "probe_video") as probe:
                with self.assertRaises(ValueError):
                    preview.inspect_local_ocr_preview(
                        movie, 0, lambda: None, preview_mode="compare-bright",
                    )
                with self.assertRaises(ValueError):
                    preview.inspect_local_ocr_preview(
                        movie, 0, lambda: None,
                        compare_make_ocr=lambda: None,
                    )
                probe.assert_not_called()

    def test_one_decode_and_one_inference_only_no_writes(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "original.mkv"
            movie.write_bytes(b"owned original unchanged")
            original = movie.read_bytes()
            made = []
            def frames(*args, **kwargs):
                made.append(kwargs)
                def generator():
                    for i in range(10):
                        yield Frame(i / 8, np.full((8, 12, 3), i, np.uint8))
                return generator()
            seen = []
            class OCR:
                def recognize(self, frame):
                    seen.append(frame.pts)
                    return OCRObservation("원본 비공개 자막", .88, frame.pts)
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=1920,height=1080,duration=80)),
                mock.patch.object(preview, "FFmpegPTSFrameSource", side_effect=frames),
            ):
                got = preview.inspect_local_ocr_preview(
                    movie, 21.18, lambda: OCR(),
                )
            self.assertEqual(len(made), 1)
            self.assertEqual(made[0]["sample_fps"], 8.0)
            self.assertEqual(made[0]["extra_input_args"],
                             ["-protocol_whitelist", "file,pipe"])
            self.assertEqual(made[0]["start_seconds"], 21)
            self.assertEqual(made[0]["preprobed_info"].width, 1920)
            self.assertEqual(made[0]["decode_timeout_seconds"], 30.0)
            self.assertEqual(got.actual_second, 21.125)
            self.assertEqual(seen, [21.125])
            self.assertEqual(got.text, "원본 비공개 자막")
            self.assertEqual(movie.read_bytes(), original)
            self.assertEqual([p.name for p in Path(td).iterdir()], ["original.mkv"])

    def test_selected_style_marker_is_returned_without_extra_inference(self):
        with tempfile.TemporaryDirectory() as td:
            media = Path(td) / "source.mkv"
            media.write_bytes(b"no real frames")
            for selected in ("bottom-two-lines", "light-caption", "caption-islands"):
                ocr = mock.Mock()
                ocr.recognize.return_value = OCRObservation(
                    "실험용 선택 자막", .91, 3.0,
                )
                with (
                    self.subTest(selected=selected),
                    mock.patch.object(preview, "probe_video",
                        return_value=mock.Mock(width=640, height=480, duration=20)),
                    mock.patch.object(preview, "FFmpegPTSFrameSource",
                        side_effect=lambda *a, **k: (item for item in [
                            Frame(0.0, np.zeros((10, 10, 3), np.uint8)),
                        ])),
                ):
                    result = preview.inspect_local_ocr_preview(
                        media, 3., lambda: ocr, preview_mode=selected,
                    )
                    self.assertEqual(result.preview_mode, selected)
                    ocr.recognize.assert_called_once()
                    self.assertEqual(result.text, "실험용 선택 자막")
            self.assertEqual(media.read_bytes(), b"no real frames")

    def test_invalid_style_rejected_before_probe_or_model(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "source.mkv"
            path.write_bytes(b"source")
            ocr = mock.Mock()
            with mock.patch.object(preview, "probe_video") as probe:
                with self.assertRaises(ValueError):
                    preview.inspect_local_ocr_preview(
                        path, 0, ocr, preview_mode="invalid-style",
                    )
                probe.assert_not_called()
                ocr.assert_not_called()

    def test_actual_ffmpeg_one_frame_preview_is_readonly(self):
        """Exercise the real pinned-style decoder on a new private synthetic clip."""
        import hashlib
        import subprocess
        from hardsub_extractor.runtime_paths import resolve_binary
        try:
            ffmpeg = resolve_binary("ffmpeg")
            resolve_binary("ffprobe")
        except (OSError, RuntimeError, FileNotFoundError):
            self.skipTest("FFmpeg not installed on this host")
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "generated.mkv"
            subprocess.run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=3",
                "-vf", "drawbox=x=12:y=42:w=24:h=10:color=white:t=fill:"
                       "enable='between(t,1.45,1.65)'",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video),
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=30, check=True)
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            observed = []
            class WhiteBoxProbe:
                def recognize(self, frame):
                    hits = int(np.count_nonzero(frame.image[:, :, 0] > 180))
                    observed.append((frame.pts, hits))
                    return OCRObservation(
                        "합성 화면" if hits > 100 else "", 0.9, frame.pts,
                    )
            # The first bounded FFprobe is sufficient for the frame
            # source. A second hidden probe should NEVER be needed.
            with mock.patch(
                "hardsub_extractor.frame_source.probe_video",
                side_effect=AssertionError("unexpected second FFprobe"),
            ) as second_probe:
                result = preview.inspect_local_ocr_preview(
                    video, 1.5, lambda: WhiteBoxProbe(),
                )
            second_probe.assert_not_called()
            self.assertEqual(len(observed), 1)
            self.assertGreater(observed[0][1], 100)
            self.assertEqual(result.text, "합성 화면")
            self.assertAlmostEqual(result.actual_second, 1.5, delta=.126)
            self.assertEqual(
                hashlib.sha256(video.read_bytes()).hexdigest(), digest,
            )
            self.assertEqual([p.name for p in Path(td).iterdir()], ["generated.mkv"])

    def test_preview_metadata_probe_is_bounded_to_25_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "source.mkv"
            movie.write_bytes(b"source unchanged")
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=640,height=480,duration=50)) as probe,
                mock.patch.object(preview, "FFmpegPTSFrameSource",
                    return_value=(frame for frame in [
                        Frame(0, np.zeros((8, 8, 3), np.uint8)),
                    ])),
            ):
                # Match the real source's closeable generator semantics.
                model = mock.Mock()
                model.recognize.return_value = OCRObservation("한 줄", .9, 0)
                result = preview.inspect_local_ocr_preview(
                    movie, 0., lambda: model,
                )
                self.assertEqual(result.text, "한 줄")
                model.recognize.assert_called_once()
                probe.assert_called_once_with(
                    str(movie.resolve()),
                    extra_input_args=["-protocol_whitelist", "file,pipe"],
                    timeout_seconds=25,
                )
            self.assertEqual(movie.read_bytes(), b"source unchanged")

    def test_changed_original_video_rejected_before_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"old video bytes")
            model = mock.Mock()
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=640,height=480,duration=12)),
                mock.patch.object(preview, "FFmpegPTSFrameSource",
                    return_value=(Frame(1.0, np.zeros((8, 8, 3), np.uint8))
                                  for _ in range(1))),
            ):
                # Mutate simulated owner data after the media probe. This
                # test only changes the isolated scratch synthetic fixture.
                original_stat = video.stat()
                def modified_ocr_factory():
                    return model()
                # A real file check between decoding and OCR must reject
                # the changed source without instantiating an OCR engine.
                with mock.patch.object(preview, "closing",
                    side_effect=lambda stream: _ModifyOnExit(stream, video)):
                    with self.assertRaisesRegex(RuntimeError, "source changed"):
                        preview.inspect_local_ocr_preview(
                            video, 1.0, modified_ocr_factory,
                        )
                model.assert_not_called()
                self.assertNotEqual(video.stat().st_size, original_stat.st_size)

    def test_no_media_frame_never_instantiates_model_or_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "episode.mkv"
            movie.write_bytes(b"video")
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=640,height=480,duration=30)),
                mock.patch.object(preview, "FFmpegPTSFrameSource",
                    side_effect=lambda *a, **k: (x for x in ())),
            ):
                model = mock.Mock()
                with self.assertRaises(RuntimeError):
                    preview.inspect_local_ocr_preview(movie, 4.0, model)
                model.assert_not_called()
            self.assertEqual(list(Path(td).iterdir()), [movie])

    def test_preview_cancel_at_empty_eof_is_not_reported_as_video_failure(self):
        from hardsub_extractor.extraction_progress import ExtractionCancelled
        import threading
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "source.mkv"
            movie.write_bytes(b"owned source")
            stop = threading.Event()
            def frames(*args, **kwargs):
                def read():
                    stop.set()
                    if False:
                        yield None
                return read()
            with (
                mock.patch.object(preview, "probe_video",
                    return_value=mock.Mock(width=640,height=480,duration=30)),
                mock.patch.object(preview, "FFmpegPTSFrameSource", side_effect=frames),
            ):
                factory = mock.Mock()
                def check():
                    if stop.is_set():
                        raise ExtractionCancelled()
                with self.assertRaises(ExtractionCancelled):
                    preview.inspect_local_ocr_preview(
                        movie, 5, factory, check_cancelled=check,
                        cancel_requested=stop.is_set,
                    )
                factory.assert_not_called()
            self.assertEqual(movie.read_bytes(), b"owned source")

    def test_bad_media_and_time_rejected_before_any_process(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "ep.mp4"
            movie.write_bytes(b"valid-ish placeholder")
            with mock.patch.object(preview, "probe_video") as probe:
                for invalid in (-1, 99999, float("nan"), True):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        preview.inspect_local_ocr_preview(movie, invalid, mock.Mock())
                probe.assert_not_called()
            with self.assertRaises(ValueError):
                preview.inspect_local_ocr_preview(Path(td) / "missing.mp4", 10, mock.Mock())

    def test_cancelled_before_model_starts(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "ep.mp4"
            movie.write_bytes(b"media")
            class Cancelled(Exception):
                pass
            with mock.patch.object(preview, "probe_video") as probe:
                with self.assertRaises(Cancelled):
                    preview.inspect_local_ocr_preview(
                        movie, 0, mock.Mock(),
                        check_cancelled=lambda: (_ for _ in ()).throw(Cancelled()),
                    )
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
