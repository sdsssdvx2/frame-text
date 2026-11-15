"""Only bounded pixel-witness timing and no-text-change behavior.

All frames are deterministic local synthetic BGR images. No real OCR model,
browser, credential, network, video file, or user data is opened.
"""
from __future__ import annotations

from unittest import TestCase, main
from unittest.mock import patch
import cv2
import numpy as np

from hardsub_extractor.models import Frame, Cue, OCRObservation
from hardsub_extractor.ocr import light_temporal as temporal


def _pixel(t: float, *, text: bool = True) -> np.ndarray:
    image = np.full((180, 420, 3), 232, np.uint8)
    cv2.rectangle(image, (30, 65), (395, 130), (65, 65, 65), -1)
    if text and 1.0 <= t < 4.0:
        cv2.putText(image, "CAPTION", (65, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return image


def _observation(t=2.5):
    return OCRObservation(
        "CAPTION", .98, t,
        {"detections": [{
            "line": 0, "text": "CAPTION", "score": .98,
            "cx": (65 + 280) / 840,
            "cy": (77 + 116) / 360,
            "width": (280 - 65) / 420,
            "height": (116 - 77) / 180,
        }]},
    )


class _OnceEngine:
    def __init__(self, observation):
        self.calls = 0
        self.observation = observation

    def recognize(self, frame):
        self.calls += 1
        return self.observation


class _SyntheticFFmpeg:
    def __init__(self, source, *, start_seconds=0.0,
                 max_duration_seconds=None, **kwargs):
        self.seek = start_seconds
        self.stop = start_seconds + (max_duration_seconds or 5.0)

    def __iter__(self):
        times = np.arange(0.0, 6.0, .125)
        for raw in times:
            actual = float(round(raw, 4))
            if self.seek <= actual < self.stop:
                yield Frame(actual - self.seek, _pixel(actual))


class LightTemporalTests(TestCase):
    def setUp(self):
        # Synthetic optical fixtures share one explicit read-only source
        # metadata record. Never invoke the real ffprobe in this unit suite.
        from hardsub_extractor.frame_source import VideoInfo
        patcher = patch.object(
            temporal, "probe_video",
            return_value=VideoInfo(420, 180, 25.0, 60.0),
        )
        self.addCleanup(patcher.stop)
        self.probe = patcher.start()

    def test_real_light_caption_timing_rescued_without_extra_ocr(self):
        raw = _observation()
        base = _OnceEngine(raw)
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        self.assertEqual(base.calls, 1)
        self.assertEqual(len(engine.witnesses), 1)
        original = Cue(1.75, 3.55, "CAPTION", .98)
        with patch.object(temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg):
            updated = engine.refine_timings([original], "synthetic", None)
        self.assertIsNot(updated[0], original)
        self.assertEqual(updated[0].text, original.text)
        self.assertEqual(updated[0].confidence, original.confidence)
        self.assertAlmostEqual(updated[0].start, 1.0, delta=.13)
        self.assertAlmostEqual(updated[0].end, 4.0, delta=.13)
        self.assertEqual(base.calls, 1)
        self.assertEqual(engine.refined_cues, 1)
        self.assertLessEqual(engine.scanned_frames, temporal.MAX_SCAN_FRAMES)

    def test_owner_cancel_during_second_pass_stops_without_model_retry(self):
        from hardsub_extractor.extraction_progress import ExtractionCancelled
        base = _OnceEngine(_observation())
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        checks = []
        def stop():
            checks.append(True)
            if len(checks) >= 7:
                # The shared metadata probe adds cancellation points;
                # interrupt only after at least one decoded optical sample.
                raise ExtractionCancelled("user stopped pixel review")
        with patch.object(temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg):
            with self.assertRaises(ExtractionCancelled):
                engine.refine_timings(
                    [Cue(1.75, 3.55, "CAPTION", .98)],
                    "synthetic", None, check_cancelled=stop,
                )
        self.assertGreater(engine.scanned_frames, 0)
        self.probe.assert_called_once()
        self.assertEqual(base.calls, 1)

    def test_optical_scan_supervises_decoder_and_uses_local_only_media(self):
        base = _OnceEngine(_observation())
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        records = []
        cancel_requested = lambda: False
        def synthetic(source, **kwargs):
            records.append((source, kwargs))
            return _SyntheticFFmpeg(source, **kwargs)
        with patch.object(temporal, "FFmpegPTSFrameSource", side_effect=synthetic):
            result = engine.refine_timings(
                [Cue(1.75, 3.55, "CAPTION", .98)],
                "synthetic", None,
                check_cancelled=lambda: None,
                cancel_requested=cancel_requested,
            )
        self.assertEqual(len(records), 1)
        source, options = records[0]
        self.assertEqual(source, "synthetic")
        self.probe.assert_called_once_with(
            "synthetic",
            extra_input_args=["-protocol_whitelist", "file,pipe"],
            timeout_seconds=25.0,
        )
        self.assertIs(options["preprobed_info"], self.probe.return_value)
        self.assertEqual(options["decode_timeout_seconds"], 90.0)
        self.assertEqual(
            options["extra_input_args"], ["-protocol_whitelist", "file,pipe"],
        )
        self.assertIs(options["cancel_requested"], cancel_requested)
        self.assertEqual(result[0].text, "CAPTION")
        self.assertEqual(base.calls, 1)

    def test_optical_probe_timeout_preserves_original_cue_without_ocr_retry(self):
        import subprocess
        base = _OnceEngine(_observation())
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        original = Cue(1.75, 3.55, "CAPTION", .98)
        with (
            patch.object(
                temporal, "probe_video",
                side_effect=subprocess.TimeoutExpired("ffprobe", 25),
            ),
            patch.object(
                temporal, "FFmpegPTSFrameSource",
                side_effect=AssertionError("no decoding after failed probe"),
            ) as decoder,
        ):
            result = engine.refine_timings([original], "synthetic", None)
        decoder.assert_not_called()
        self.assertIs(result[0], original)
        self.assertEqual(engine.failed_decoder_batches, 1)
        self.assertEqual(engine.skipped_cues, 1)
        self.assertEqual(engine.decoder_launches, 0)
        self.assertEqual(engine.refined_cues, 0)
        self.assertEqual(base.calls, 1)

    def test_time_index_selects_exact_original_candidate_set_and_order(self):
        # The new binary-search window must NOT lose a real OCR witness at
        # its boundaries or alter equal-PTS tie-breaking and cue quality.
        times = [5.2, 1.2, 2.5, 2.5, 3.7, 1.9, 50.0, .2]
        witnesses = [
            temporal._capture_witness(
                Frame(t, _pixel(2.5)), _observation(t),
            )
            for t in times
        ]
        self.assertTrue(all(w is not None for w in witnesses))
        index, pts = temporal._witness_time_index(witnesses)
        for cue in (
            Cue(1.75, 3.55, "CAPTION", .98),
            Cue(4.0, 5.8, "CAPTION", .98),
            Cue(40.0, 41.0, "CAPTION", .98),
            Cue(.0, .2, "CAPTION", .98),
        ):
            with self.subTest(cue=(cue.start, cue.end)):
                nearby = temporal._witnesses_near_cue(cue, index, pts)
                expected_candidates = [
                    w for w in witnesses
                    if abs(w.pts - max(cue.start, min(w.pts, cue.end))) < .60
                ]
                actual_candidates = [
                    w for w in nearby
                    if abs(w.pts - max(cue.start, min(w.pts, cue.end))) < .60
                ]
                self.assertEqual(actual_candidates, expected_candidates)
                self.assertEqual(
                    temporal._choose_anchors(cue, nearby),
                    temporal._choose_anchors(cue, witnesses),
                )

    def test_distant_optical_windows_use_single_shared_metadata_probe(self):
        original = Cue(1.75, 3.55, "CAPTION", .98)
        later = Cue(33.25, 34.80, "CAPTION", .98)
        base = _OnceEngine(_observation())
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        additional = temporal._capture_witness(
            Frame(34.0, _pixel(2.5)), _observation(34.0),
        )
        self.assertIsNotNone(additional)
        engine.witnesses.append(additional)
        with patch.object(
            temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg,
        ):
            result = engine.refine_timings(
                [original, later], "synthetic", None,
            )
        self.probe.assert_called_once()
        self.assertEqual(engine.decoder_launches, 2)
        self.assertEqual(result[0].text, original.text)
        self.assertIs(result[1], later)
        self.assertEqual(base.calls, 1)

    def test_witness_memory_limit_retains_late_episode_evidence(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)
        base = _OnceEngine(_observation(2.5))
        engine = temporal.LightTemporalWitnessEngine(base)
        image = _pixel(2.5)
        # Shrink the *real* production capacity for a long-video simulation
        # without running the heavy OCR engine or 100,000 image masks.
        with (
            patch.object(temporal, "MAX_WITNESSES", 24),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(seed, pts=frame.pts),
            ) as capture,
        ):
            for ordinal in range(1000):
                engine.recognize(Frame(float(ordinal), image))
                self.assertLessEqual(len(engine.witnesses), 24)
        kept_pts = [w.pts for w in engine.witnesses]
        self.assertEqual(base.calls, 1000)
        self.assertGreater(engine.witness_sample_stride, 1)
        self.assertGreater(engine.witness_compactions, 0)
        self.assertEqual(len(engine._witness_run_ids), len(engine.witnesses))
        self.assertEqual(len(engine.witnesses), len(engine._witness_ordinals))
        self.assertEqual(kept_pts, sorted(set(kept_pts)))
        self.assertEqual(kept_pts[0], 0.)
        self.assertGreater(kept_pts[-1], 900.)
        self.assertLess(capture.call_count, 1000)
        self.assertTrue(temporal._choose_anchors(
            Cue(0., .5, "CAPTION", .98), engine.witnesses,
        ))
        self.assertTrue(temporal._choose_anchors(
            Cue(959.8, 960.2, "CAPTION", .98), engine.witnesses,
        ))

    def test_many_short_dialogues_retain_middle_episode_witnesses_after_compaction(self):
        from dataclasses import replace
        from hardsub_extractor.ocr.base import comparison_key
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class RapidDialogue:
            def recognize(self, frame):
                return OCRObservation(
                    f"caption-{int(frame.pts)}", .98, frame.pts,
                )

        engine = temporal.LightTemporalWitnessEngine(RapidDialogue())
        image = _pixel(2.5)
        # Frequent entirely different dialogue is an extreme stress case;
        # no finite 128-mask budget can retain all 4,000 transitions.
        # Require usable evidence distributed across ALL video quarters,
        # rather than nearly only its first and final minutes.
        with (
            patch.object(temporal, "MAX_WITNESSES", 128),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(
                    seed, pts=frame.pts,
                    key=comparison_key(obs.text), text=obs.text,
                ),
            ),
        ):
            for second in range(4000):
                engine.recognize(Frame(float(second), image))
                self.assertLessEqual(len(engine.witnesses), 128)
        quarters = [
            sum(
                start <= witness.pts < start + 1000
                for witness in engine.witnesses
            )
            for start in (0, 1000, 2000, 3000)
        ]
        self.assertGreaterEqual(min(quarters), 5)
        self.assertEqual(engine.witnesses[-1].pts, 3999.)
        self.assertEqual(
            len(engine._witness_run_ids), len(engine.witnesses),
        )
        self.assertGreater(engine.witness_compactions, 0)

    def test_repeated_cue_samples_are_evicted_before_unique_caption_starts(self):
        from dataclasses import replace
        from hardsub_extractor.ocr.base import comparison_key
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class RepeatedRuns:
            def recognize(self, frame):
                return OCRObservation(
                    f"caption-{int(frame.pts // 2)}", .98, frame.pts,
                )

        engine = temporal.LightTemporalWitnessEngine(RepeatedRuns())
        image = _pixel(2.5)
        with (
            patch.object(temporal, "MAX_WITNESSES", 32),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(
                    seed, pts=frame.pts,
                    key=comparison_key(obs.text), text=obs.text,
                ),
            ),
        ):
            # 10 distinct 2-second captions; 0.5-second witness cadence
            # yields four samples each, forcing compaction with 32 slots.
            for frame_index in range(200):
                second = frame_index / 10.
                engine.recognize(Frame(second, image))
        saved_dialogue = {w.text for w in engine.witnesses}
        self.assertEqual(
            saved_dialogue,
            {f"caption-{i}" for i in range(10)},
        )
        self.assertLessEqual(len(engine.witnesses), 32)
        self.assertGreater(engine.witness_compactions, 0)
        self.assertEqual(
            len(engine.witnesses), len(engine._witness_run_ids),
        )

    def test_below_witness_limit_keeps_exact_existing_sample_sequence(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)
        base = _OnceEngine(_observation(2.5))
        engine = temporal.LightTemporalWitnessEngine(base)
        with (
            patch.object(temporal, "MAX_WITNESSES", 24),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(seed, pts=frame.pts),
            ),
        ):
            for ordinal in range(12):
                engine.recognize(Frame(float(ordinal), _pixel(2.5)))
        self.assertEqual([w.pts for w in engine.witnesses], list(map(float, range(12))))
        self.assertEqual(engine.witness_sample_stride, 1)
        self.assertEqual(engine.witness_compactions, 0)
        self.assertEqual(base.calls, 12)

    def test_dense_94_minute_caption_runs_keep_all_new_phrases_under_budget(self):
        from dataclasses import replace
        from hardsub_extractor.ocr.base import comparison_key
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class SyntheticReader:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                i = round(frame.pts * 8)
                text = f"dialogue-{i // 18}" if i % 18 < 16 else ""
                return OCRObservation(text, .98, frame.pts)

        base = SyntheticReader()
        engine = temporal.LightTemporalWitnessEngine(base)
        image = _pixel(2.5)
        with patch.object(
            temporal, "_capture_witness",
            side_effect=lambda frame, obs: replace(
                seed, pts=frame.pts,
                key=comparison_key(obs.text), text=obs.text,
            ),
        ):
            for i in range(45000):
                engine.recognize(Frame(i / 8.0, image))
        counts = {}
        for witness in engine.witnesses:
            counts[witness.text] = counts.get(witness.text, 0) + 1
        self.assertEqual(base.calls, 45000)
        self.assertEqual(len(counts), 2500)
        self.assertEqual(len(engine.witnesses), 10000)
        self.assertEqual(engine.witness_compactions, 0)
        self.assertEqual(engine.witness_sample_stride, 1)
        self.assertTrue(all(count == 4 for count in counts.values()))
        self.assertGreater(engine.witness_duplicate_skips, 20000)
        self.assertIn("dialogue-2499", counts)

    def test_new_short_caption_is_not_dropped_by_global_sparse_stride(self):
        from dataclasses import replace
        from hardsub_extractor.ocr.base import comparison_key
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class ChangingReader:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                return OCRObservation(
                    f"dialogue-{self.calls}", .98, frame.pts,
                )

        base = ChangingReader()
        engine = temporal.LightTemporalWitnessEngine(base)
        image = _pixel(2.5)
        with (
            patch.object(temporal, "MAX_WITNESSES", 24),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(
                    seed, pts=frame.pts,
                    key=comparison_key(obs.text), text=obs.text,
                ),
            ),
        ):
            for i in range(400):
                engine.recognize(Frame(i / 8.0, image))
                self.assertLessEqual(len(engine.witnesses), 24)
        self.assertEqual(base.calls, 400)
        self.assertGreater(engine.witness_compactions, 0)
        self.assertGreater(engine.witness_sample_stride, 1)
        self.assertEqual(engine.witnesses[-1].text, "dialogue-400")
        self.assertEqual(len(engine.witnesses), len(engine._witness_ordinals))

    def test_same_words_after_blank_frame_start_new_optical_run(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class ReappearingReader:
            def recognize(self, frame):
                text = "CAPTION" if frame.pts != .125 else ""
                return OCRObservation(text, .98, frame.pts)

        engine = temporal.LightTemporalWitnessEngine(ReappearingReader())
        with patch.object(
            temporal, "_capture_witness",
            side_effect=lambda frame, obs: replace(seed, pts=frame.pts),
        ):
            for pts in (0., .125, .25):
                engine.recognize(Frame(pts, _pixel(2.5)))
        self.assertEqual([w.pts for w in engine.witnesses], [0., .25])
        self.assertEqual(engine.witness_duplicate_skips, 0)

    def test_pending_first_valid_caption_witness_survives_sparse_sampling(self):
        from dataclasses import replace
        from hardsub_extractor.ocr.base import comparison_key
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class NewCaption:
            def recognize(self, frame):
                return OCRObservation("NEW CAPTION", .98, frame.pts)

        engine = temporal.LightTemporalWitnessEngine(NewCaption())
        # Model a real memory compaction at a sparse stride. The first
        # physical video frame of a new caption has an invalid glyph mask;
        # the next frame is valid but its ordinal is NOT on the stride.
        engine.witness_sample_stride = 4
        engine._witness_seen = 5
        engine._last_observed_key = "OLD CAPTION"
        engine._last_captured_key = "OLD CAPTION"
        engine._current_witness_run = 1
        attempted = []
        def capture(frame, obs):
            attempted.append(frame.pts)
            if len(attempted) == 1:
                return None
            return replace(
                seed, pts=frame.pts, text=obs.text,
                key=comparison_key(obs.text),
            )
        with patch.object(temporal, "_capture_witness", side_effect=capture):
            engine.recognize(Frame(10.0, _pixel(2.5)))
            self.assertTrue(engine._pending_run_anchor)
            engine.recognize(Frame(10.125, _pixel(2.5)))
        self.assertEqual(attempted, [10.0, 10.125])
        self.assertEqual([w.pts for w in engine.witnesses], [10.125])
        self.assertEqual(engine._witness_ordinals, [6])
        self.assertEqual(engine._witness_run_ids, [2])
        self.assertFalse(engine._pending_run_anchor)

    def test_relocated_caption_failed_box_keeps_one_pending_run_until_valid(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class MovedWords:
            def recognize(self, frame):
                observed = _observation(frame.pts)
                observed.metadata["detections"][0]["cx"] += .14
                return observed

        engine = temporal.LightTemporalWitnessEngine(MovedWords())
        engine.witness_sample_stride = 4
        engine._witness_seen = 5
        engine._last_observed_key = "CAPTION"
        engine._last_captured_key = "CAPTION"
        engine._current_witness_run = 1
        engine._last_captured_layout = temporal._ocr_box_layout(_observation())
        attempts = []
        def capture(frame, obs):
            attempts.append(frame.pts)
            return None if len(attempts) == 1 else replace(seed, pts=frame.pts)
        with patch.object(temporal, "_capture_witness", side_effect=capture):
            engine.recognize(Frame(10., _pixel(2.5)))
            self.assertTrue(engine._pending_run_anchor)
            engine.recognize(Frame(10.125, _pixel(2.5)))
        self.assertEqual(attempts, [10., 10.125])
        self.assertEqual([w.pts for w in engine.witnesses], [10.125])
        # It remains a single new visual run, not a new run per failed box.
        self.assertEqual(engine._witness_run_ids, [2])
        self.assertFalse(engine._pending_run_anchor)

    def test_sparse_stride_preserves_moving_same_text_after_budget_compaction(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class MovingAfterCap:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                observation = _observation(frame.pts)
                if frame.pts >= 9.0:
                    observation.metadata["detections"][0]["cx"] += .14
                return observation

        reader = MovingAfterCap()
        engine = temporal.LightTemporalWitnessEngine(reader)
        with (
            patch.object(temporal, "MAX_WITNESSES", 8),
            patch.object(
                temporal, "_capture_witness",
                side_effect=lambda frame, obs: replace(seed, pts=frame.pts),
            ),
        ):
            for second in range(10):
                engine.recognize(Frame(float(second), _pixel(2.5)))
                self.assertLessEqual(len(engine.witnesses), 8)

        self.assertEqual(reader.calls, 10)
        self.assertEqual(engine.witness_sample_stride, 2)
        # After compaction, the non-stride ordinal 9 still has a scene cut.
        # The old code read its new mask then discarded it at append.
        self.assertEqual(engine._witness_ordinals[-1], 9)
        self.assertEqual(engine.witnesses[-1].pts, 9.0)
        self.assertEqual(engine.witness_duplicate_skips, 0)
        self.assertGreater(engine.witness_compactions, 0)

    def test_same_words_moving_with_scene_are_not_skipped_as_duplicates(self):
        from dataclasses import replace
        seed = temporal._capture_witness(
            Frame(2.5, _pixel(2.5)), _observation(2.5),
        )
        self.assertIsNotNone(seed)

        class MovingCaption:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                observed = _observation(frame.pts)
                # Small motion/jitter should be treated as a duplicate, but
                # a 14%-of-screen horizontal shift changes the pixel anchor.
                observed.metadata["detections"][0]["cx"] += (
                    0 if self.calls == 1 else
                    .01 if self.calls == 2 else .14
                )
                return observed

        engine = temporal.LightTemporalWitnessEngine(MovingCaption())
        with patch.object(
            temporal, "_capture_witness",
            side_effect=lambda frame, obs: replace(seed, pts=frame.pts),
        ):
            for t in (0., .125, .25):
                engine.recognize(Frame(t, _pixel(2.5)))
        self.assertEqual([w.pts for w in engine.witnesses], [0., .25])
        self.assertEqual(engine.witness_duplicate_skips, 1)

    def test_two_independent_matching_short_cjk_glyphs_enable_optical_timing(self):
        # Script labels are synthetic OCR observations on repeatable bright
        # glyph pixels, NOT a claim that cv2 can render real Chinese/Japanese.
        for phrase in ("안녕", "はい", "你好"):
            with self.subTest(phrase=phrase):
                obs1 = _observation(2.25)
                obs1.text = phrase
                obs1.metadata["detections"][0]["text"] = phrase
                obs2 = _observation(2.75)
                obs2.text = phrase
                obs2.metadata["detections"][0]["text"] = phrase
                engine = temporal.LightTemporalWitnessEngine(_OnceEngine(obs1))
                engine.recognize(Frame(2.25, _pixel(2.5)))
                self.assertEqual(len(engine.witnesses), 1)
                second = temporal._capture_witness(
                    Frame(2.75, _pixel(2.5)), obs2,
                )
                self.assertIsNotNone(second)
                engine.witnesses.append(second)
                cue = Cue(1.75, 3.55, phrase, .98)
                anchors = temporal._choose_anchors(cue, engine.witnesses)
                self.assertEqual(len(anchors), 2)
                self.assertEqual([a.pts for a in anchors], [2.25, 2.75])
                with patch.object(
                    temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg,
                ):
                    result = engine.refine_timings([cue], "synthetic", None)
                self.assertEqual(result[0].text, phrase)
                self.assertEqual(result[0].confidence, cue.confidence)
                self.assertAlmostEqual(result[0].start, 1.0, delta=.13)
                self.assertAlmostEqual(result[0].end, 4.0, delta=.13)
                self.assertEqual(engine.inner.calls, 1)

    def test_real_recognize_path_keeps_two_short_cjk_witnesses_025s_apart(self):
        # The former test manually appended a second witness, skipping the
        # real 0.50s duplicate gate. Verify full recognize() -> witness
        # selection -> optical timing proposal without that shortcut.
        for text in ("안녕", "はい", "你好"):
            with self.subTest(text=text):
                observed = _observation(2.25)
                observed.text = text
                observed.metadata["detections"][0]["text"] = text
                reader = _OnceEngine(observed)
                engine = temporal.LightTemporalWitnessEngine(reader)
                engine.recognize(Frame(2.25, _pixel(2.5)))
                engine.recognize(Frame(2.50, _pixel(2.5)))
                self.assertEqual(reader.calls, 2)
                self.assertEqual(
                    [w.pts for w in engine.witnesses],
                    [2.25, 2.50],
                )
                self.assertEqual(engine.witness_duplicate_skips, 0)
                cue = Cue(1.75, 3.55, text, .98)
                self.assertEqual(
                    len(temporal._choose_anchors(cue, engine.witnesses)), 2
                )
                with patch.object(
                    temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg
                ):
                    corrected = engine.refine_timings([cue], "synthetic", None)
                self.assertEqual(engine.refined_cues, 1)
                self.assertAlmostEqual(corrected[0].start, 1., delta=.13)
                self.assertAlmostEqual(corrected[0].end, 4., delta=.13)
                self.assertEqual(corrected[0].text, text)
                self.assertEqual(corrected[0].confidence, cue.confidence)
                self.assertEqual(reader.calls, 2)  # no OCR rerun

    def test_two_short_cjk_observations_do_not_spawn_unbounded_duplicate_masks(self):
        observed = _observation(2.25)
        observed.text = "안녕"
        observed.metadata["detections"][0]["text"] = "안녕"
        engine = temporal.LightTemporalWitnessEngine(_OnceEngine(observed))
        for pts in (2.25, 2.375, 2.50, 2.625, 2.75, 2.875):
            engine.recognize(Frame(pts, _pixel(2.5)))
        self.assertEqual([w.pts for w in engine.witnesses], [2.25, 2.50])
        self.assertGreaterEqual(engine.witness_duplicate_skips, 3)
        self.assertEqual(engine._current_run_saved_witnesses, 2)

    def test_short_latin_ui_and_single_cjk_keep_existing_half_second_dedup(self):
        for text in ("OK", "TV", "아", "中", "CAPTION"):
            with self.subTest(text=text):
                observed = _observation(2.25)
                observed.text = text
                observed.metadata["detections"][0]["text"] = text
                engine = temporal.LightTemporalWitnessEngine(_OnceEngine(observed))
                for pts in (2.25, 2.50):
                    engine.recognize(Frame(pts, _pixel(2.5)))
                # Existing actual glyph capture rejects 1-character
                # detections even before any temporal deduplication.
                expected = 0 if len(text) == 1 else 1
                self.assertEqual(len(engine.witnesses), expected)

    def test_short_cjk_second_witness_survives_sparse_compaction_stride(self):
        observed = _observation(2.25)
        observed.text = "你好"
        observed.metadata["detections"][0]["text"] = "你好"
        engine = temporal.LightTemporalWitnessEngine(_OnceEngine(observed))
        engine.witness_sample_stride = 4
        engine._witness_seen = 5
        engine.recognize(Frame(2.25, _pixel(2.5)))
        engine.recognize(Frame(2.50, _pixel(2.5)))
        self.assertEqual([w.pts for w in engine.witnesses], [2.25, 2.50])
        self.assertEqual(engine._witness_ordinals, [5, 6])
        self.assertEqual(engine._witness_run_ids, [1, 1])
        cue = Cue(1.75, 3.55, "你好", .98)
        self.assertEqual(len(temporal._choose_anchors(cue, engine.witnesses)), 2)

    def test_one_short_cjk_glyph_witness_never_changes_time(self):
        observed = _observation(2.5)
        observed.text = "你好"
        observed.metadata["detections"][0]["text"] = "你好"
        engine = temporal.LightTemporalWitnessEngine(_OnceEngine(observed))
        engine.recognize(Frame(2.5, _pixel(2.5)))
        original = Cue(1.75, 3.55, "你好", .98)
        self.assertEqual(
            temporal._choose_anchors(original, engine.witnesses), (),
        )
        with patch.object(
            temporal, "FFmpegPTSFrameSource",
            side_effect=AssertionError("no decode without two CJK witnesses"),
        ):
            corrected = engine.refine_timings([original], "synthetic", None)
        self.assertIs(corrected[0], original)
        self.probe.assert_not_called()

    def test_short_latin_label_and_one_character_remain_unmodified(self):
        for phrase in ("OK", "TV", "HP", "中", "아"):
            with self.subTest(phrase=phrase):
                obs = _observation(2.25)
                obs.text = phrase
                obs.metadata["detections"][0]["text"] = phrase
                first = temporal._capture_witness(
                    Frame(2.25, _pixel(2.5)), obs,
                )
                later = temporal._capture_witness(
                    Frame(2.75, _pixel(2.5)), obs,
                )
                # The one-character observation is rejected by the
                # existing real _capture_witness minimum glyph length.
                if first is None or later is None:
                    continue
                self.assertEqual(
                    temporal._choose_anchors(
                        Cue(1.75, 3.55, phrase, .98), [first, later]
                    ), (),
                )

    def test_short_cjk_same_text_but_wrong_pixels_or_time_cannot_corrobate(self):
        from dataclasses import replace
        obs = _observation(2.25)
        obs.text = "你好"
        obs.metadata["detections"][0]["text"] = "你好"
        one = temporal._capture_witness(
            Frame(2.25, _pixel(2.5)), obs,
        )
        two = temporal._capture_witness(
            Frame(2.75, _pixel(2.5)), obs,
        )
        self.assertIsNotNone(one)
        self.assertIsNotNone(two)
        cue = Cue(1.75, 3.55, "你好", .98)
        other_pixels = replace(
            two, boxes=tuple(
                replace(
                    box,
                    packed_mask=bytes(v ^ 255 for v in box.packed_mask)
                ) for box in two.boxes
            ),
        )
        self.assertEqual(
            temporal._choose_anchors(cue, [one, other_pixels]), (),
        )
        duplicate_time = replace(two, pts=2.35)
        self.assertEqual(
            temporal._choose_anchors(cue, [one, duplicate_time]), (),
        )
        different_word = replace(two, text="您好", key="您好")
        self.assertEqual(
            temporal._choose_anchors(cue, [one, different_word]), (),
        )

    def test_no_verified_witness_preserves_original_exactly(self):
        base = _OnceEngine(OCRObservation("CAPTION", .98, 2.5, None))
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        original = Cue(1.75, 3.55, "CAPTION", .98)
        with patch.object(temporal, "FFmpegPTSFrameSource", side_effect=AssertionError("unexpected second decode")):
            updated = engine.refine_timings([original], "synthetic", None)
        self.assertIs(updated[0], original)
        self.assertEqual(engine.refined_cues, 0)

    def test_visual_presence_not_inferred_from_blank_page(self):
        anchor = temporal._capture_witness(Frame(2.5, _pixel(2.5)), _observation())
        self.assertIsNotNone(anchor)
        self.assertGreaterEqual(
            temporal._presence(Frame(2.5, _pixel(2.5)), anchor),
            temporal.MIN_MATCH,
        )
        self.assertLess(
            temporal._presence(Frame(2.5, _pixel(0.0)), anchor),
            temporal.MIN_MATCH,
        )

    def test_bright_background_noise_does_not_hide_actual_glyph_strokes(self):
        anchor = temporal._capture_witness(Frame(2.5, _pixel(2.5)), _observation())
        noisy = _pixel(2.5)
        # Same white subtitle strokes remain while additional unrelated
        # moving highlight pixels appear in the letter's wider OCR box.
        cv2.rectangle(noisy, (65, 77), (279, 84), (239,239,239), -1)
        self.assertGreaterEqual(
            temporal._presence(Frame(2.5, noisy), anchor),
            temporal.MIN_MATCH,
        )
        white_page = np.full_like(noisy, 255)
        self.assertLess(
            temporal._presence(Frame(2.5, white_page), anchor),
            temporal.MIN_MATCH,
        )

    def test_wrong_dialogue_cannot_supply_anchor(self):
        witness = temporal._capture_witness(Frame(2.5, _pixel(2.5)), _observation())
        self.assertIsNone(temporal._choose_anchor(Cue(1.0, 4.0, "DIFFERENT", .99), [witness]))

    def test_adjacent_cue_overlap_rejected(self):
        base = _OnceEngine(_observation())
        engine = temporal.LightTemporalWitnessEngine(base)
        engine.recognize(Frame(2.5, _pixel(2.5)))
        existing = Cue(1.75, 3.55, "CAPTION", .98)
        next_cue = Cue(3.8, 4.7, "NEXT WORDS", .98)
        with patch.object(temporal, "FFmpegPTSFrameSource", _SyntheticFFmpeg):
            result = engine.refine_timings([existing, next_cue], "synthetic", None)
        self.assertIs(result[0], existing)
        self.assertIs(result[1], next_cue)
        self.assertEqual(engine.refined_cues, 0)

    def test_scene_change_shifted_caption_uses_two_geometry_witnesses(self):
        def shifted(t):
            image = np.full((180, 420, 3), 232, np.uint8)
            cv2.rectangle(image, (30, 65), (395, 130), (65, 65, 65), -1)
            if 1.0 <= t < 4.0:
                x = 65 if t < 2.5 else 125
                cv2.putText(image, "CAPTION", (x, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (255,255,255), 2, cv2.LINE_AA)
            return image

        early = temporal._capture_witness(
            Frame(1.5, shifted(1.5)), _observation(1.5),
        )
        late_result = _observation(3.0)
        late_result.metadata["detections"][0]["cx"] += 60 / 420
        late = temporal._capture_witness(
            Frame(3.0, shifted(3.0)), late_result,
        )
        cue = Cue(1.75, 3.55, "CAPTION", .98)
        anchors = temporal._choose_anchors(cue, [early, late])
        self.assertEqual(len(anchors), 2)
        frames = [
            Frame(round(n * .125, 4), shifted(n * .125))
            for n in range(46)
        ]
        revised = temporal._single_refined(cue, anchors, frames)
        self.assertIsNotNone(revised)
        self.assertAlmostEqual(revised[0], 1.0, delta=.13)
        self.assertAlmostEqual(revised[1], 4.0, delta=.13)

    def test_long_true_blank_between_repeat_captions_must_not_be_filled(self):
        def split_image(t):
            frame = np.full((180, 420, 3), 232, np.uint8)
            cv2.rectangle(frame, (30, 65), (395, 130), (65,65,65), -1)
            if (1.0 <= t < 2.0 or 3.0 <= t < 4.0):
                cv2.putText(frame, "CAPTION", (65,112),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,
                            (255,255,255),2,cv2.LINE_AA)
            return frame
        early = temporal._capture_witness(
            Frame(1.5, split_image(1.5)), _observation(1.5),
        )
        late = temporal._capture_witness(
            Frame(3.5, split_image(3.5)), _observation(3.5),
        )
        cue = Cue(1.75,3.55,"CAPTION",.98)
        anchors = temporal._choose_anchors(cue,[early,late])
        self.assertEqual(len(anchors),2)
        frames = [Frame(float(k*.125),split_image(k*.125)) for k in range(46)]
        revised = temporal._single_refined(cue, anchors, frames)
        # Two OCR witnesses are insufficient to prove that a caption was
        # visible during a full second with NO matching glyph pixels.
        self.assertIsNone(revised)

    def test_shared_decode_for_adjacent_cues_does_not_repeat_ffmpeg(self):
        def image_at(t):
            image = np.full((180, 420, 3), 235, np.uint8)
            cv2.rectangle(image, (30, 65), (395, 130), (62,62,62), -1)
            if 1.0 <= t < 2.5:
                cv2.putText(image, "FIRST", (65,112),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,
                            (255,255,255),2,cv2.LINE_AA)
            elif 2.5 <= t < 4.0:
                cv2.putText(image, "SECOND", (65,112),
                            cv2.FONT_HERSHEY_SIMPLEX,1.1,
                            (255,255,255),2,cv2.LINE_AA)
            return image

        def observed(text, pts):
            o = _observation(pts)
            o.text = text
            o.metadata["detections"][0]["text"] = text
            return o

        first = temporal._capture_witness(
            Frame(1.5, image_at(1.5)), observed("FIRST CAPTION", 1.5))
        second = temporal._capture_witness(
            Frame(3.0, image_at(3.0)), observed("SECOND CAPTION", 3.0))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        class Source:
            opens = 0
            def __init__(self, source, *, start_seconds=0.0,
                         max_duration_seconds=None, **kwargs):
                self.seek = start_seconds
                self.stop = start_seconds + (max_duration_seconds or 1.)
                Source.opens += 1
            def __iter__(self):
                for raw in np.arange(0, 6, .125):
                    t = float(round(raw, 4))
                    if self.seek <= t < self.stop:
                        yield Frame(t - self.seek, image_at(t))

        engine = temporal.LightTemporalWitnessEngine(_OnceEngine(_observation()))
        engine.witnesses = [first, second]
        original = [
            Cue(1.125, 2.375, "FIRST CAPTION", .98),
            Cue(2.625, 3.875, "SECOND CAPTION", .98),
        ]
        with patch.object(temporal, "FFmpegPTSFrameSource", Source):
            refined = engine.refine_timings(original, "synthetic", None)
        self.assertEqual(Source.opens, 1, "adjacent cues should share one decoder")
        self.assertEqual(engine.decoder_launches, 1)
        self.assertEqual(len(refined), 2)
        self.assertEqual([q.text for q in refined], [q.text for q in original])
        self.assertLessEqual(engine.scanned_frames, temporal.MAX_SCAN_FRAMES)
        self.assertLessEqual(engine.scanned_frames, 50)

    def test_witness_masks_use_compact_bits_not_full_frames(self):
        item = temporal._capture_witness(Frame(2.5, _pixel(2.5)), _observation())
        self.assertIsNotNone(item)
        self.assertTrue(item.boxes)
        for box in item.boxes:
            self.assertIsInstance(box.packed_mask, bytes)
            self.assertEqual(len(box.packed_mask), temporal.BOX_H * temporal.BOX_W // 8)

    def test_missing_frame_evidence_keeps_existing_srt_cue(self):
        witness = temporal._capture_witness(Frame(2.5, _pixel(2.5)), _observation())
        cue = Cue(2.0, 2.4, "CAPTION", .98)
        frames = [Frame(x * .125, _pixel(0.0)) for x in range(40)]
        self.assertIsNone(temporal._single_refined(cue, witness, frames))


if __name__ == "__main__":
    main()
