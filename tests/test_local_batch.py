"""Local 1–50 video batch: no-network routing, provenance and no-overwrite."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import local_batch


class LocalBatchTests(unittest.TestCase):
    def make_source(self, root: Path, name: str, content: bytes = b"fakevideo") -> Path:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def test_authored_ogv_is_explicitly_accepted_without_implicit_directory_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            authored = self.make_source(root, "日本語字幕.ogv")
            tracks = local_batch.load_local_batch(
                [authored], None, root / "output",
            )
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].source, authored)
            self.assertIn("日本語字幕", tracks[0].out_base.name)
            self.assertFalse((root / "output").exists())
            # A generic OGG audio file is not a video shortcut.
            audio = self.make_source(root, "audio.ogg")
            with self.assertRaises(ValueError):
                local_batch.load_local_batch([audio], None, root / "out")

    def test_selected_roi_reaches_hard_video_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "video.mp4")
            items = local_batch.load_local_batch([video], None, root / "output")
            with (mock.patch.object(local_batch, "acquire_local_embedded_text", return_value=None),
                  mock.patch.object(local_batch, "acquire_local_sidecar_text", return_value=None),
                  mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=None),
                  mock.patch.object(local_batch, "extract_hard_subtitles",
                                    return_value=SimpleNamespace(cue_count=2)) as extracted):
                result = local_batch.run_local_batch(
                    items, ocr_factory=lambda: mock.Mock(), region_mode="bottom25",
                )
            self.assertEqual(result[0].mode, "hard_ocr")
            roi = extracted.call_args.kwargs["roi"]
            self.assertEqual((roi.x, roi.y, roi.w, roi.h), (.04, .73, .92, .25))
            self.assertFalse((root / "output").exists())

    def test_native_roi_mismatch_rejected_before_any_source_or_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "video.mkv")
            items = local_batch.load_local_batch([video], None, root / "output")
            with mock.patch.object(local_batch, "acquire_local_embedded_text") as source:
                with self.assertRaises(ValueError):
                    local_batch.run_local_batch(
                        items, ocr_factory=mock.Mock(),
                        native_timing_factory=lambda: (Path("native"), "sha"),
                        region_mode="lower80",
                    )
                source.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_one_to_fifty_videos_with_readable_korean_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = [self.make_source(root, f"한글 회차 {i:02}.mkv")
                     for i in range(50)]
            loaded = local_batch.load_local_batch(files, None, root / "out")
            self.assertEqual(len(loaded), 50)
            self.assertTrue(all("한글" in x.out_base.name for x in loaded))
            self.assertEqual(len({x.out_base.name for x in loaded}), 50)

    def test_reject_51_before_any_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [self.make_source(root, f"ep{i:02}.mp4") for i in range(51)]
            with self.assertRaises(ValueError):
                local_batch.load_local_batch(videos, None, root / "out")
            self.assertFalse((root / "out").exists())

    def test_file_list_is_relative_to_file_location_with_comment_and_bom(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "assets/episode.mp4")
            lst = root / "episodes.txt"
            lst.write_text(
                "\ufeff# list created by user\nassets/episode.mp4\n\n"
                "# another comment\n", encoding="utf-8",
            )
            result = local_batch.load_local_batch([], lst, root / "outputs")
            self.assertEqual([x.source for x in result], [video])

    def test_same_stem_across_folders_produces_distinct_result_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self.make_source(root, "part-A/episode.mp4")
            b = self.make_source(root, "part-B/episode.mp4")
            result = local_batch.load_local_batch([a, b], None, root / "out")
            self.assertNotEqual(result[0].out_base, result[1].out_base)
            self.assertTrue(result[0].out_base.name.startswith("episode-"))
            self.assertTrue(result[1].out_base.name.startswith("episode-"))

    def test_dotted_episode_names_keep_distinct_digest_in_real_result_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            one = self.make_source(root, "episode.01.mkv")
            two = self.make_source(root, "episode.02.mkv")
            items = local_batch.load_local_batch(
                [one, two], None, root / "out",
            )
            self.assertEqual(len(items), 2)
            self.assertTrue(all(not item.out_base.suffix for item in items))
            outputs = {
                item.out_base.with_suffix(extension)
                for item in items for extension in (".srt", ".ass")
            }
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(
                len(path.stem.rsplit("-", 1)[-1]) == 16
                for path in outputs
            ))
            self.assertIn("episode_01-", items[0].out_base.name)
            self.assertIn("episode_02-", items[1].out_base.name)
            self.assertFalse((root / "out").exists())

    def test_duplicate_video_is_deduplicated_without_reordering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self.make_source(root, "first.mkv")
            b = self.make_source(root, "second.mkv")
            lst = root / "list.txt"
            lst.write_text(str(a) + "\n" + str(b) + "\n" + str(a), "utf-8")
            paths = local_batch.load_local_batch([b, a], lst, root / "out")
            self.assertEqual([p.source for p in paths], [b, a])

    def test_reject_unsupported_nonexistent_empty_and_symlink_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = [
                root / "missing.mp4",
                self.make_source(root, "empty.mp4", b""),
                self.make_source(root, "script.py"),
            ]
            target = self.make_source(root, "valid.mp4")
            alias = root / "alias.mp4"
            try:
                alias.symlink_to(target)
                bad.append(alias)
            except OSError:
                pass
            for source in bad:
                with self.subTest(name=source.name), self.assertRaises(ValueError):
                    local_batch.load_local_batch([source], None, root / "out")

    def test_output_file_blocks_source_work_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = self.make_source(root, "ep.mp4")
            items = local_batch.load_local_batch([src], None, root / "out")
            items[0].out_base.parent.mkdir()
            raw = items[0].out_base.with_suffix(".srt")
            raw.write_bytes(b"previous file byte exact")
            with mock.patch.object(local_batch, "acquire_local_embedded_text") as embedded:
                result = local_batch.run_local_batch(
                    items, ocr_factory=lambda: self.fail("OCR invoked"),
                )
            embedded.assert_not_called()
            self.assertEqual(raw.read_bytes(), b"previous file byte exact")
            self.assertEqual(result[0].mode, "existing_preserved")

    def test_embedded_first_sidecar_second_and_ocr_lazy_across_three(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [self.make_source(root, f"episode{i}.mkv")
                      for i in range(3)]
            items = local_batch.load_local_batch(videos, None, root / "out")
            calls = []
            def embedded(src, out, **kw):
                calls.append(("embedded", src.name))
                if src.name == "episode0.mkv":
                    return SimpleNamespace(mode="local-embedded-text", cue_count=4)
                return None
            def sidecar(src, out, **kw):
                calls.append(("sidecar", src.name))
                if src.name == "episode1.mkv":
                    return SimpleNamespace(mode="local-sidecar-text", cue_count=6)
                return None
            created = []
            def ocr_factory():
                created.append(True)
                return object()
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", side_effect=embedded),
                mock.patch.object(local_batch, "acquire_local_sidecar_text", side_effect=sidecar),
                mock.patch.object(local_batch, "extract_hard_subtitles",
                                  return_value=SimpleNamespace(cue_count=8)) as hard,
            ):
                result = local_batch.run_local_batch(items, ocr_factory=ocr_factory)
            self.assertEqual([x.mode for x in result],
                             ["local-embedded-text", "local-sidecar-text", "hard_ocr"])
            self.assertEqual([x.cues for x in result], [4, 6, 8])
            self.assertEqual(len(created), 1)
            self.assertEqual(hard.call_count, 1)
            self.assertEqual(
                calls,
                [("embedded", "episode0.mkv"),
                 ("embedded", "episode1.mkv"), ("sidecar", "episode1.mkv"),
                 ("embedded", "episode2.mkv"), ("sidecar", "episode2.mkv")],
            )
            self.assertTrue(hard.call_args.kwargs["streaming"])
            self.assertNotIn("sample_fps", hard.call_args.kwargs)
            self.assertTrue(hard.call_args.kwargs["monitor"].exclusive_outputs)

    def test_caption_islands_opt_in_only_for_hard_ocr_and_model_reused(self):
        from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [self.make_source(root, f"ep{i}.mkv") for i in range(2)]
            items = local_batch.load_local_batch(videos, None, root / "out")
            originals = []
            def factory():
                model = object()
                originals.append(model)
                return model
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", return_value=None),
                mock.patch.object(local_batch, "acquire_local_sidecar_text", return_value=None),
                mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=None),
                mock.patch.object(local_batch, "extract_hard_subtitles",
                                  return_value=SimpleNamespace(cue_count=2)) as hard,
            ):
                result = local_batch.run_local_batch(
                    items, ocr_factory=factory, caption_islands=True,
                )
            self.assertEqual([r.mode for r in result], ["hard_ocr", "hard_ocr"])
            self.assertEqual(len(originals), 1)
            self.assertEqual(hard.call_count, 2)
            wrappers = [call.args[1] for call in hard.call_args_list]
            self.assertTrue(all(isinstance(w, CaptionIslandEngine) for w in wrappers))
            self.assertIsNot(wrappers[0], wrappers[1])
            self.assertTrue(all(w.engine is originals[0] for w in wrappers))

    def test_caption_islands_conflicts_rejected_before_any_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = local_batch.load_local_batch(
                [self.make_source(root, "ep.mkv")], None, root / "out",
            )
            for option in (
                {"caption_islands": True, "focus_bottom_two_lines": True},
                {"caption_islands": True, "light_caption_temporal": True},
                {"caption_islands": True, "native_timing_factory": lambda: None},
                {"caption_islands": "yes"},
            ):
                with self.subTest(option=option), self.assertRaises(ValueError):
                    local_batch.run_local_batch(
                        item, ocr_factory=lambda: self.fail("model invoked"), **option,
                    )
            self.assertFalse((root / "out").exists())

    def test_bottom_two_line_opt_in_passed_only_to_visual_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "no-soft-subtitle.mkv")
            items = local_batch.load_local_batch([video], None, root / "out")
            with (
                mock.patch.object(
                    local_batch, "acquire_local_embedded_text", return_value=None
                ),
                mock.patch.object(
                    local_batch, "acquire_local_sidecar_text", return_value=None
                ),
                mock.patch.object(
                    local_batch, "acquire_local_embedded_image", return_value=None
                ) as image,
                mock.patch.object(
                    local_batch, "extract_hard_subtitles",
                    return_value=SimpleNamespace(cue_count=2),
                ) as visual,
            ):
                result = local_batch.run_local_batch(
                    items, ocr_factory=lambda: object(),
                    focus_bottom_two_lines=True,
                )
            self.assertEqual(result[0].mode, "hard_ocr")
            image.assert_called_once()
            self.assertTrue(visual.call_args.kwargs["focus_bottom_two_lines"])

    def test_bottom_two_line_opt_in_rejected_with_native_detector(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "source.mkv")
            items = local_batch.load_local_batch([video], None, root / "out")
            native = mock.Mock()
            with self.assertRaises(ValueError):
                local_batch.run_local_batch(
                    items, ocr_factory=lambda: self.fail("OCR invoked"),
                    native_timing_factory=native,
                    focus_bottom_two_lines=True,
                )
            native.assert_not_called()

    def test_light_caption_timing_source_first_and_separate_witnesses_per_video(self):
        from functools import partial
        from hardsub_extractor.ocr.light_text import LightCaptionEngine
        from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [self.make_source(root, f"clip-{i}.mkv") for i in range(3)]
            items = local_batch.load_local_batch(videos, None, root / "out")

            def original(video, out, **kwargs):
                if video == videos[0]:
                    return SimpleNamespace(mode="local-embedded-text", cue_count=3)
                return None

            created_models = []
            def new_model():
                model = mock.Mock()
                created_models.append(model)
                return model

            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text",
                                  side_effect=original),
                mock.patch.object(local_batch, "acquire_local_sidecar_text",
                                  return_value=None),
                mock.patch.object(local_batch, "acquire_local_embedded_image",
                                  return_value=None),
                mock.patch.object(local_batch, "extract_hard_subtitles",
                                  return_value=SimpleNamespace(cue_count=2)) as visual,
            ):
                result = local_batch.run_local_batch(
                    items, ocr_factory=new_model, light_caption_temporal=True,
                )
            self.assertEqual([row.mode for row in result],
                             ["local-embedded-text", "hard_ocr", "hard_ocr"])
            self.assertEqual((len(created_models), visual.call_count), (1, 2))
            wrappers = [call.args[1] for call in visual.call_args_list]
            self.assertTrue(all(isinstance(e, LightTemporalWitnessEngine) for e in wrappers))
            self.assertIsNot(wrappers[0], wrappers[1])
            self.assertTrue(all(isinstance(e.inner, LightCaptionEngine) for e in wrappers))
            self.assertIs(wrappers[0].inner.inner, wrappers[1].inner.inner)
            for call, engine in zip(visual.call_args_list, wrappers):
                refiner = call.kwargs["timing_refiner"]
                self.assertIsInstance(refiner, partial)
                self.assertIs(refiner.func.__self__, engine)
                self.assertTrue(call.kwargs["monitor"].exclusive_outputs)
            self.assertEqual([r.optical_timing_refined for r in result], [0, 0, 0])
            self.assertEqual([r.optical_witness_stride for r in result], [1, 1, 1])

    def test_light_caption_timing_rejects_conflicting_or_invalid_options(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = local_batch.load_local_batch(
                [self.make_source(root, "one.mkv")], None, root / "out",
            )
            for options in (
                {"light_caption_temporal": "yes"},
                {"light_caption_temporal": True, "focus_bottom_two_lines": True},
                {"light_caption_temporal": True, "native_timing_factory": lambda: None},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    local_batch.run_local_batch(
                        items, ocr_factory=lambda: self.fail("no OCR expected"),
                        **options,
                    )

    def test_bitmap_original_before_visual_ocr_on_source_first_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "image-track.mkv")
            items = local_batch.load_local_batch([video], None, root / "out")
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", return_value=None) as embedded,
                mock.patch.object(local_batch, "acquire_local_sidecar_text", return_value=None) as sidecar,
                mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=SimpleNamespace(
                    mode="local-embedded-image-ocr", cue_count=3,
                )) as bitmap,
                mock.patch.object(local_batch, "extract_hard_subtitles") as visual,
            ):
                result = local_batch.run_local_batch(
                    items, language="en",
                    ocr_factory=lambda: self.fail("video visual OCR model initialized"),
                )
            embedded.assert_called_once()
            sidecar.assert_called_once()
            bitmap.assert_called_once()
            self.assertEqual(bitmap.call_args.args, (video, items[0].out_base))
            self.assertEqual(bitmap.call_args.kwargs["language"], "en")
            self.assertIsNone(bitmap.call_args.kwargs["check_cancelled"])
            self.assertTrue(callable(bitmap.call_args.kwargs["ocr_factory"]))
            visual.assert_not_called()
            self.assertEqual((result[0].mode, result[0].cues),
                             ("local-embedded-image-ocr", 3))

    def test_partial_pgs_bitmap_does_not_count_as_complete_episode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [
                self.make_source(root, "bitmap.mkv"),
                self.make_source(root, "text.mkv"),
            ]
            items = local_batch.load_local_batch(videos, None, root / "out")
            def embedded(video, out, **kw):
                return (
                    SimpleNamespace(mode="local-embedded-text", cue_count=2)
                    if video == videos[1] else None
                )
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text",
                                  side_effect=embedded),
                mock.patch.object(local_batch, "acquire_local_sidecar_text",
                                  return_value=None),
                mock.patch.object(local_batch, "acquire_local_embedded_image",
                                  return_value=SimpleNamespace(
                                      mode="local-embedded-image-ocr",
                                      cue_count=1, missed_events=2,
                                  )) as image,
                mock.patch.object(local_batch, "extract_hard_subtitles") as visual,
            ):
                result = local_batch.run_local_batch(
                    items, language="en",
                    ocr_factory=lambda: self.fail("unexpected visual OCR"),
                    native_timing_factory=lambda: self.fail("unexpected native download"),
                )
            self.assertEqual([x.mode for x in result], [
                "local-embedded-image-ocr-partial", "local-embedded-text",
            ])
            self.assertEqual([x.missed_events for x in result], [2, 0])
            self.assertEqual([x.cues for x in result], [1, 2])
            self.assertEqual(image.call_count, 1)
            visual.assert_not_called()

    def test_failed_episode_does_not_block_subsequent_video(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = local_batch.load_local_batch(
                [self.make_source(root, f"{i}.mp4") for i in range(3)],
                None, root / "out",
            )
            events = []
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text",
                                  side_effect=[RuntimeError("private filename"), None, None]),
                mock.patch.object(local_batch, "acquire_local_sidecar_text",
                                  return_value=SimpleNamespace(
                                      mode="local-sidecar-text", cue_count=2)),
            ):
                results = local_batch.run_local_batch(
                    items, ocr_factory=lambda: self.fail("model not needed"),
                    on_result=events.append,
                )
            self.assertEqual([x.mode for x in results],
                             ["failed", "local-sidecar-text", "local-sidecar-text"])
            self.assertEqual(results[0].failure_class, "RuntimeError")
            self.assertNotIn("private filename", repr(results))
            self.assertEqual(results, events)

    def test_native_timing_only_for_missing_originals_and_only_one_official_setup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sources = [self.make_source(root, f"ep{i}.mkv") for i in range(5)]
            items = local_batch.load_local_batch(sources, None, root / "output")
            setups = []
            models = []
            def embedded(video, out, **kw):
                return SimpleNamespace(mode="local-embedded-text", cue_count=2) if video == sources[0] else None
            def sidecar(video, out, **kw):
                return SimpleNamespace(mode="local-sidecar-text", cue_count=3) if video == sources[1] else None
            def images(video, out, **kw):
                return SimpleNamespace(mode="local-embedded-image-ocr", cue_count=4) if video == sources[2] else None
            def official():
                setups.append("one checked official archive for this batch")
                return Path("private-verified-official.exe"), "d" * 64
            fake_report = SimpleNamespace(
                recognized_intervals=2, missed_intervals=1,
                exported_cues=3, split_source_intervals=1,
                split_cue_numbers=(2,),
                short_frame_rescue_attempted=2,
                short_frame_rescue_recognized=1,
                short_frame_rescue_deferred=3,
                short_frame_rescue_errors=0,
            )
            def native_call(*args, **kwargs):
                # The *real* timed OCR reader lazily calls the factory
                # only after valid timing detection. Simulate that here.
                kwargs["ocr_factory"]()
                return SimpleNamespace(report=fake_report)
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", side_effect=embedded),
                mock.patch.object(local_batch, "acquire_local_sidecar_text", side_effect=sidecar),
                mock.patch.object(local_batch, "acquire_local_embedded_image", side_effect=images),
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           side_effect=native_call) as native,
                mock.patch.object(local_batch, "extract_hard_subtitles") as original,
            ):
                results = local_batch.run_local_batch(
                    items, language="ja",
                    ocr_factory=lambda: models.append("model") or object(),
                    native_timing_factory=official,
                    native_split_opt_in=True,
                )
            self.assertEqual([x.mode for x in results], [
                "local-embedded-text", "local-sidecar-text",
                "local-embedded-image-ocr", "hard_ocr_native_timing_partial",
                "hard_ocr_native_timing_partial",
            ])
            self.assertEqual([x.missed_events for x in results], [0, 0, 0, 1, 1])
            self.assertEqual([x.cues for x in results[-2:]], [3, 3])
            self.assertEqual(
                [x.split_cue_numbers for x in results[-2:]],
                [(2,), (2,)],
            )
            self.assertTrue(native.call_args.kwargs["split_stable_text"])
            self.assertEqual(
                [(r.short_frame_rescue_attempted,
                  r.short_frame_rescue_recognized,
                  r.short_frame_rescue_deferred)
                 for r in results[-2:]],
                [(2, 1, 3), (2, 1, 3)],
            )
            self.assertEqual(setups, ["one checked official archive for this batch"])
            self.assertEqual(len(models), 1)
            self.assertEqual(native.call_count, 2)
            self.assertEqual(native.call_args.kwargs["pinned_executable_sha256"], "d"*64)
            self.assertTrue(callable(native.call_args.kwargs["ocr_factory"]))
            original.assert_not_called()

    def test_native_full_coverage_is_reported_complete_not_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = local_batch.load_local_batch(
                [self.make_source(root, "one.mkv")], None, root/"output",
            )
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", return_value=None),
                mock.patch.object(local_batch, "acquire_local_sidecar_text", return_value=None),
                mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=None),
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           return_value=SimpleNamespace(report=SimpleNamespace(
                               recognized_intervals=3, missed_intervals=0,
                           ))),
            ):
                result = local_batch.run_local_batch(
                    item, ocr_factory=lambda: object(),
                    native_timing_factory=lambda: (Path("exe"), "a"*64),
                )
            self.assertEqual(result[0].mode, "hard_ocr_native_timing")
            self.assertEqual(result[0].cues, 3)
            self.assertEqual(result[0].missed_events, 0)

    def test_native_optional_setup_failure_once_and_other_source_subtitles_continue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sources = [self.make_source(root, f"ep{i}.mkv") for i in range(3)]
            items = local_batch.load_local_batch(sources, None, root / "output")
            count = []
            def embedded(video, out, **kw): return None
            def sidecar(video, out, **kw):
                return SimpleNamespace(mode="local-sidecar-text", cue_count=5) if video == sources[1] else None
            def no_tool():
                count.append(1)
                raise RuntimeError("optional helper unavailable offline")
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text", side_effect=embedded),
                mock.patch.object(local_batch, "acquire_local_sidecar_text", side_effect=sidecar),
                mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=None),
                mock.patch.object(local_batch, "extract_hard_subtitles") as fallback,
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf") as native,
            ):
                results = local_batch.run_local_batch(
                    items, ocr_factory=lambda: self.fail("unexpected OCR"),
                    native_timing_factory=no_tool,
                )
            self.assertEqual([x.mode for x in results],
                             ["failed", "local-sidecar-text", "failed"])
            self.assertEqual(len(count), 1)
            fallback.assert_not_called()
            native.assert_not_called()

    def test_native_cancellation_propagates_without_touching_later_episodes(self):
        from hardsub_extractor.extraction_progress import ExtractionCancelled
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = local_batch.load_local_batch(
                [self.make_source(root, "ep1.mkv"),
                 self.make_source(root, "ep2.mkv")], None, root / "output",
            )
            seen = []
            def native_cancel():
                raise ExtractionCancelled("cancelled before native download")
            with (
                mock.patch.object(local_batch, "acquire_local_embedded_text",
                                  return_value=None) as embedded,
                mock.patch.object(local_batch, "acquire_local_sidecar_text", return_value=None),
                mock.patch.object(local_batch, "acquire_local_embedded_image", return_value=None),
            ):
                with self.assertRaises(ExtractionCancelled):
                    local_batch.run_local_batch(
                        items, ocr_factory=lambda: self.fail("unexpected OCR"),
                        native_timing_factory=native_cancel,
                        on_result=seen.append,
                    )
            self.assertEqual(embedded.call_count, 1)
            self.assertEqual(seen, [])

    def test_native_option_ignored_for_existing_preserved_subtitle_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.make_source(root, "episode.mkv")
            items = local_batch.load_local_batch([source], None, root / "results")
            items[0].out_base.parent.mkdir()
            original = items[0].out_base.with_suffix(".srt")
            original.write_bytes(b"USER OWNED SUBTITLE")
            with mock.patch.object(local_batch, "acquire_local_embedded_text") as embedded:
                results = local_batch.run_local_batch(
                    items, ocr_factory=lambda: self.fail("OCR invoked"),
                    native_timing_factory=lambda: self.fail("download invoked"),
                )
            embedded.assert_not_called()
            self.assertEqual(results[0].mode, "existing_preserved")
            self.assertEqual(original.read_bytes(), b"USER OWNED SUBTITLE")

    def test_cancellation_propagates_instead_of_being_marked_failed(self):
        class Cancelled(Exception):
            pass
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = local_batch.load_local_batch(
                [self.make_source(root, "1.mp4")], None, root / "out",
            )
            with self.assertRaises(Cancelled):
                local_batch.run_local_batch(
                    items, ocr_factory=lambda: object(),
                    check_cancelled=lambda: (_ for _ in ()).throw(Cancelled()),
                )

    def test_preflight_rejects_invalid_middle_source_before_creating_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            one = self.make_source(root, "one.mp4")
            lst = root / "list.txt"
            lst.write_text("one.mp4\nmissing.mp4\n", "utf-8")
            with self.assertRaises(ValueError):
                local_batch.load_local_batch([], lst, root / "out")
            self.assertFalse((root / "out").exists())
            self.assertTrue(one.is_file())

    def test_existing_non_directory_destination_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.make_source(root, "clip.mkv")
            occupied = root / "occupied"
            occupied.write_text("do not overwrite", "utf-8")
            with self.assertRaises(ValueError):
                local_batch.load_local_batch([video], None, occupied)
            self.assertEqual(occupied.read_text("utf-8"), "do not overwrite")

    def test_output_collision_validation_before_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = local_batch.load_local_batch(
                [self.make_source(root, "ep.mp4")], None, root / "out",
            )[0]
            with self.assertRaises(ValueError):
                local_batch.run_local_batch(
                    [item, item], ocr_factory=lambda: self.fail("bad collision"),
                )


if __name__ == "__main__":
    unittest.main()
