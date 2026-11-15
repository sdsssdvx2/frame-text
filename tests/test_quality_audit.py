from __future__ import annotations

import json
import math
import random
import unittest

from hardsub_extractor.models import Cue
from hardsub_extractor.quality_audit import (
    AuditOptions, audit_subtitles, _edit_distance, _align, _align_sparse,
)


def C(start, end, text):
    return Cue(start, end, text)


class QualityAuditTests(unittest.TestCase):
    def test_candidate_only_does_not_claim_accuracy(self):
        data = audit_subtitles([C(1, 2, "안녕하세요")])
        self.assertFalse(data["accuracy_verified"])
        self.assertFalse(data["contains_subtitle_text"])
        self.assertFalse(data["reference_comparison"]["reference_supplied"])
        self.assertEqual(data["candidate"]["cue_count"], 1)
        self.assertNotIn("안녕하세요", json.dumps(data, ensure_ascii=False))

    def test_identical_reference_has_zero_edit_error(self):
        cues = [C(1, 2, "안녕하세요"), C(3, 4.2, "오늘은 좋은 날!")]
        result = audit_subtitles(cues, reference=list(cues))["reference_comparison"]
        self.assertEqual(result["one_to_one_aligned_cues"], 2)
        self.assertEqual(result["text_exact_match_count_aligned_only"], 2)
        self.assertEqual(result["character_error_rate_aligned_only"], 0.0)
        self.assertEqual(result["reference_alignment_fraction"], 1.0)

    def test_hangul_one_character_substitution_is_measured(self):
        candidate = [C(2, 3, "사랑해")]
        reference = [C(2, 3, "사랑햐")]
        result = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(result["one_to_one_aligned_cues"], 1)
        self.assertEqual(result["text_exact_match_count_aligned_only"], 0)
        self.assertEqual(result["character_error_rate_aligned_only"], 0.3333)

    def test_reference_constant_shift_must_be_explicit(self):
        candidate = [C(101, 102, "안녕")]
        reference = [C(1, 2, "안녕")]
        without = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        with_shift = audit_subtitles(candidate, reference=reference, options=AuditOptions(reference_shift_seconds=100))["reference_comparison"]
        self.assertEqual(without["one_to_one_aligned_cues"], 0)
        self.assertEqual(with_shift["one_to_one_aligned_cues"], 1)
        self.assertEqual(with_shift["median_absolute_start_error_seconds_aligned_only"], 0.0)

    def test_one_missing_middle_cue_is_not_hidden_by_index_pairing(self):
        candidate = [C(1, 2, "가"), C(5, 6, "다")]
        reference = [C(1, 2, "가"), C(3, 4, "나"), C(5, 6, "다")]
        result = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(result["one_to_one_aligned_cues"], 2)
        self.assertEqual(result["unmatched_reference_count"], 1)
        self.assertEqual(result["unmatched_reference_cue_numbers_first_30"], [2])
        self.assertEqual(result["character_error_rate_aligned_only"], 0.0)

    def test_one_extra_middle_cue_remains_unmatched(self):
        candidate = [C(1, 2, "가"), C(3, 4, "추가"), C(5, 6, "다")]
        reference = [C(1, 2, "가"), C(5, 6, "다")]
        result = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(result["one_to_one_aligned_cues"], 2)
        self.assertEqual(result["unmatched_candidate_cue_numbers_first_30"], [2])
        self.assertTrue(result["comparison_is_not_ground_truth_accuracy"])

    def test_split_candidate_is_explicitly_not_claimed_false_positive(self):
        candidate = [C(1, 2, "첫 구절"), C(2, 3, "둘째 구절")]
        reference = [C(1, 3, "첫 구절 둘째 구절")]
        result = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(result["one_to_one_aligned_cues"], 1)
        self.assertEqual(result["unmatched_candidate_count"], 1)
        self.assertEqual(result["unmatched_candidate_with_any_time_overlap_count"], 1)
        self.assertTrue(result["one_to_many_split_merge_may_affect_unmatched_counts"])

    def test_large_gap_is_warning_not_verified_missing_subtitle(self):
        candidate = [C(1, 2, "처음"), C(150, 152, "나중")]
        result = audit_subtitles(candidate)
        self.assertEqual(result["candidate"]["warning_counts"]["long_gap_review"], 1)
        self.assertTrue(result["candidate"]["warnings_are_not_verified_ocr_errors"])

    def test_fast_very_short_and_nonkorean_are_review_only(self):
        candidate = [C(1, 1.1, "ABCDEFGHIJKLMNO")]
        result = audit_subtitles(candidate)["candidate"]
        self.assertEqual(result["warning_counts"]["fast_reading_review"], 1)
        self.assertEqual(result["warning_counts"]["very_short_duration_review"], 1)
        self.assertEqual(result["warning_counts"]["no_hangul_review"], 1)


    def test_oversized_ocr_cue_is_review_only_and_preserves_text(self):
        huge = "字幕" * 301
        candidate = [C(1, 4, huge), C(5, 7, "正常字幕")]
        before = [x.text for x in candidate]
        result = audit_subtitles(candidate)["candidate"]
        self.assertEqual(
            result["warning_counts"]["oversized_ocr_cue_review"], 1,
        )
        self.assertEqual(
            result["review_locations"]["oversized_ocr_cue_review"],
            [{"cue": 1, "start_seconds": 1}],
        )
        self.assertEqual(before, [x.text for x in candidate])

    def test_many_physical_lines_are_reviewed_without_length_guessing(self):
        text = "\n".join(f"line {i}" for i in range(9))
        result = audit_subtitles([C(1, 3, text)])["candidate"]
        self.assertEqual(
            result["warning_counts"]["oversized_ocr_cue_review"], 1,
        )

    def test_repeated_overlay_row_marks_review_without_rewriting_text(self):
        lines = [
            C(1, 2, "固定電視台台標文字\n第一句對話"),
            C(2.5, 3.5, "固定電視台台標文字\n第二句對話"),
            C(4, 5, "固定電視台台標文字\n第三句對話"),
            C(6, 7, "一段真正的字幕"),
        ]
        before = [(x.start, x.end, x.text) for x in lines]
        report = audit_subtitles(lines, options=AuditOptions(
            requested_language="zh",
        ))
        candidate = report["candidate"]
        self.assertEqual(
            candidate["warning_counts"]["repeated_multiline_overlay_review"], 1,
        )
        self.assertEqual(
            candidate["review_locations"]["repeated_multiline_overlay_review"],
            [{"cue": 1, "start_seconds": 1}],
        )
        self.assertEqual(before, [(x.start, x.end, x.text) for x in lines])
        self.assertFalse(report["accuracy_verified"])
        self.assertNotIn("固定電視台台標文字", json.dumps(report, ensure_ascii=False))

    def test_short_repeated_labels_do_not_trigger_overlay_warning(self):
        cues = [
            C(1, 2, "固定電視台台標\n第一句對話"),
            C(2.5, 3.5, "固定電視台台標\n第二句對話"),
            C(4, 5, "固定電視台台標\n第三句對話"),
        ]
        result = audit_subtitles(cues, options=AuditOptions(
            requested_language="zh",
        ))["candidate"]
        self.assertNotIn(
            "repeated_multiline_overlay_review", result["warning_counts"],
        )

    def test_repeated_bottom_row_is_review_only(self):
        lines = [
            C(1, 2, "첫 번째 대사\nCHANNEL COPYRIGHT"),
            C(2.2, 3, "두 번째 대사\nCHANNEL COPYRIGHT"),
            C(3.3, 4, "세 번째 대사\nCHANNEL COPYRIGHT"),
        ]
        counts = audit_subtitles(lines)["candidate"]["warning_counts"]
        self.assertEqual(counts["repeated_multiline_overlay_review"], 1)

    def test_true_repeated_dialogue_and_two_cue_runs_are_not_overlay_proof(self):
        single = [C(i, i + 1, "안녕하세요") for i in (1, 3, 5, 7)]
        same_two = [C(i, i + 1, "매번 같은 대사\n둘째 줄도 같은 대사")
                    for i in (1, 3, 5)]
        only_two = [
            C(1, 2, "반복되는 문장입니다\n첫째"),
            C(2.5, 3.5, "반복되는 문장입니다\n둘째"),
        ]
        distant = [
            C(1, 2, "반복되는 문장입니다\n첫째"),
            C(30, 31, "반복되는 문장입니다\n둘째"),
            C(60, 61, "반복되는 문장입니다\n셋째"),
        ]
        for cues in (single, same_two, only_two, distant):
            with self.subTest(size=len(cues)):
                report = audit_subtitles(cues)["candidate"]
                self.assertNotIn(
                    "repeated_multiline_overlay_review", report["warning_counts"],
                )

    def test_warning_location_pagination_does_not_truncate_long_episode(self):
        cues = [
            C(float(i * 2), float(i * 2 + 1), "English scene text")
            for i in range(40)
        ]
        report = audit_subtitles(cues, options=AuditOptions(
            requested_language="zh", max_review_locations=2,
            review_locations_after_cue=35,
        ))
        item = report["candidate"]
        self.assertEqual(item["warning_counts"]["no_han_review"], 40)
        self.assertEqual(item["review_locations_after_cue"], 35)
        self.assertEqual(
            [entry["cue"] for entry in item["review_locations"]["no_han_review"]],
            [36, 37],
        )
        self.assertFalse(report["accuracy_verified"])
        self.assertNotIn("English scene text", json.dumps(report))

    def test_warning_location_cursor_type_and_budget_are_fail_closed(self):
        cue = C(1, 2, "English only")
        for cursor in (-1, 10001, 1.0, True, "12", None):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                audit_subtitles([cue], options=AuditOptions(
                    review_locations_after_cue=cursor,
                ))

    def test_review_location_limit_does_not_hide_warning_count(self):
        cues = []
        for block in range(2):
            base = block * 40
            for i in range(3):
                cues.append(C(base + i*2, base + i*2 + 1,
                              "반복 화면 중첩 문구\n다른 대사 " + str(i)))
        item = audit_subtitles(cues, options=AuditOptions(
            max_review_locations=1,
        ))["candidate"]
        self.assertEqual(
            item["warning_counts"]["repeated_multiline_overlay_review"], 2,
        )
        self.assertEqual(
            len(item["review_locations"]["repeated_multiline_overlay_review"]), 1,
        )

    def test_duplicate_and_overlap_warning_without_text_leak(self):
        candidate = [C(1, 3, "비밀문구"), C(2.9, 4, "비밀문구")]
        result = audit_subtitles(candidate)["candidate"]
        self.assertEqual(result["warning_counts"]["nearby_duplicate_text_review"], 1)
        self.assertEqual(result["warning_counts"]["overlapping_cues_review"], 1)
        self.assertNotIn("비밀문구", json.dumps(result, ensure_ascii=False))

    def test_empty_punctuation_comparison_is_not_duplicate(self):
        result = audit_subtitles([C(1, 2, "!?"), C(2, 3, "!!")])["candidate"]
        self.assertNotIn("nearby_duplicate_text_review", result["warning_counts"])

    def test_invalid_timeline_fails_closed(self):
        for invalid in ([C(2, 1, "안녕")], [C(float("nan"), 2, "안녕")], [C(2, math.inf, "안녕")]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "candidate_invalid_timeline"):
                    audit_subtitles(invalid)

    def test_out_of_order_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "candidate_out_of_order"):
            audit_subtitles([C(5, 6, "나"), C(2, 3, "가")])

    def test_bad_tolerances_or_shifts_fail_closed(self):
        for opt in (AuditOptions(alignment_tolerance_seconds=-1), AuditOptions(reference_shift_seconds=math.nan), AuditOptions(max_review_locations=-1)):
            with self.subTest(opt=opt):
                with self.assertRaisesRegex(ValueError, "invalid_options"):
                    audit_subtitles([C(1, 2, "가")], options=opt)

    def test_sparse_tie_break_matches_original_dense_alignment(self):
        # The small-input implementation is immutable; this cross-check
        # ensures the long-input path does not silently change ambiguity
        # handling when multiple overlapping cues admit equal scores.
        rng = random.Random(4219)
        for case in range(120):
            cn = rng.randrange(1, 18)
            rn = rng.randrange(1, 18)
            c_starts = sorted(rng.randrange(0, 35) * 0.5 for _ in range(cn))
            r_starts = sorted(rng.randrange(0, 35) * 0.5 for _ in range(rn))
            candidate = [
                C(start, start + rng.choice((.5, 1., 1.5, 3.)),
                  rng.choice(("한글", "漢字", "Hello", "日本語")))
                for start in c_starts
            ]
            reference = [
                C(start, start + rng.choice((.5, 1., 1.5, 3.)),
                  rng.choice(("한글", "漢字", "Hello", "日本語")))
                for start in r_starts
            ]
            options = AuditOptions(reference_shift_seconds=rng.choice((0., .25, -.5)))
            with self.subTest(case=case):
                self.assertEqual(
                    _align_sparse(candidate, reference, options),
                    _align(candidate, reference, options),
                )

    def test_1500_cue_full_episode_reference_no_longer_rejected(self):
        # 2.25 million Cartesian pairs previously failed before comparing
        # even a perfect long movie. Only local temporal neighbors should
        # enter the sparse matcher. No text enters the returned JSON report.
        cues = [C(i * 2., i * 2. + 1., f"비공개 대사 {i}")
                for i in range(1500)]
        original = [(c.start, c.end, c.text) for c in cues]
        report = audit_subtitles(cues, reference=list(cues))
        comparison = report["reference_comparison"]
        self.assertEqual(comparison["one_to_one_aligned_cues"], 1500)
        self.assertEqual(comparison["text_exact_match_count_aligned_only"], 1500)
        self.assertEqual(comparison["unmatched_candidate_count"], 0)
        self.assertEqual(comparison["unmatched_reference_count"], 0)
        self.assertEqual(comparison["character_error_rate_aligned_only"], 0.0)
        self.assertFalse(report["accuracy_verified"])
        self.assertNotIn("비공개 대사", json.dumps(report, ensure_ascii=False))
        self.assertEqual([(c.start, c.end, c.text) for c in cues], original)

    def test_long_movie_one_missing_middle_cue_is_reported_not_hidden(self):
        reference = [C(i * 2., i * 2. + 1., f"대사 {i}")
                     for i in range(1500)]
        candidate = [cue for i, cue in enumerate(reference) if i != 750]
        report = audit_subtitles(candidate, reference=reference)
        data = report["reference_comparison"]
        self.assertEqual(data["one_to_one_aligned_cues"], 1499)
        self.assertEqual(data["unmatched_reference_count"], 1)
        self.assertEqual(data["unmatched_reference_cue_numbers_first_30"], [751])
        self.assertEqual(data["unmatched_candidate_count"], 0)
        self.assertEqual(data["character_error_rate_aligned_only"], 0.0)
        self.assertFalse(report["accuracy_verified"])

    def test_1500_cue_mismatched_reference_stays_unmatched_without_n2(self):
        # No time overlap: do not invent aligned captions for adjacent
        # episode segments even when the full Cartesian size is enormous.
        candidate = [C(i * 3., i * 3. + 1., "COPY") for i in range(1500)]
        reference = [C(i * 3. + 1.5, i * 3. + 2.5, "COPY") for i in range(1500)]
        compared = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(compared["one_to_one_aligned_cues"], 0)
        self.assertEqual(compared["unmatched_candidate_count"], 1500)
        self.assertEqual(compared["unmatched_reference_count"], 1500)
        self.assertEqual(compared["unmatched_candidate_with_any_time_overlap_count"], 0)
        self.assertEqual(compared["unmatched_reference_with_any_time_overlap_count"], 0)

    def test_reference_pair_budget_prevents_accidental_quadratic_exhaustion(self):
        one = C(1, 2, "가")
        with self.assertRaisesRegex(ValueError, "pair_budget_exceeded"):
            audit_subtitles([one] * 1500, reference=[one] * 1500)

    def test_editorial_difference_not_equivalent_to_ocr_accuracy(self):
        candidate = [C(1, 3, "정말 좋아!")]
        reference = [C(1, 3, "참 마음에 들어")]
        result = audit_subtitles(candidate, reference=reference)
        self.assertEqual(result["reference_comparison"]["one_to_one_aligned_cues"], 1)
        self.assertGreater(result["reference_comparison"]["character_error_rate_aligned_only"], 0)
        self.assertFalse(result["accuracy_verified"])

    def test_extremely_long_cue_cannot_steal_short_reference(self):
        candidate = [C(1.0, 50.0, "긴 영상 구간")]
        reference = [C(1.0, 2.0, "짧은 대사")]
        data = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(data["one_to_one_aligned_cues"], 0)
        self.assertEqual(data["unmatched_reference_count"], 1)
        self.assertEqual(data["unmatched_candidate_with_any_time_overlap_count"], 1)

    def test_adjacent_nonoverlapping_cues_are_not_aligned(self):
        candidate = [C(1.0, 1.7, "첫 자막")]
        reference = [C(1.7, 2.4, "다음 자막")]
        data = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(data["one_to_one_aligned_cues"], 0)

    def test_partial_overlap_with_sensible_durations_still_aligns(self):
        candidate = [C(1.0, 3.0, "같은 자막")]
        reference = [C(2.0, 4.0, "같은 자막")]
        data = audit_subtitles(candidate, reference=reference)["reference_comparison"]
        self.assertEqual(data["one_to_one_aligned_cues"], 1)

    def test_oversized_cue_text_is_rejected_before_edit_distance(self):
        with self.assertRaisesRegex(ValueError, "candidate_text_budget_exceeded"):
            audit_subtitles([C(1, 2, "가" * 4097)], reference=[C(1, 2, "나")])

    def test_unreasonably_large_cue_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate_cue_budget_exceeded"):
            audit_subtitles([C(1, 2, "가")] * 10001)

    def test_empty_text_is_rejected_for_audit(self):
        with self.assertRaisesRegex(ValueError, "reference_empty_text"):
            audit_subtitles([C(1, 2, "가")], reference=[C(1, 2, "")])

    def test_edit_distance_is_exact_and_handles_insertions(self):
        self.assertEqual(_edit_distance("abc", "ab"), 1)
        self.assertEqual(_edit_distance("안녕", "안녕하세요"), 3)
        self.assertEqual(_edit_distance("", "안녕"), 2)

    def test_does_not_mutate_original_subtitles(self):
        cue = C(1, 2, "첫 줄\n둘째 줄")
        audit_subtitles([cue], reference=[C(1, 2, "첫 줄 둘째 줄")])
        self.assertEqual(cue.text, "첫 줄\n둘째 줄")


    def test_chosen_multilingual_scripts_do_not_generate_korean_false_warnings(self):
        examples = [
            ("ko", "가나다", "no_hangul_review"),
            ("en", "Where are you going?", "no_latin_review"),
            ("ja", "今日は学校へ行きます", "no_japanese_script_review"),
            ("ja", "学校生活", "no_japanese_script_review"),
            ("zh", "今天天气很好", "no_han_review"),
            ("zh-Hans", "今天晚上一起去看电影吧", "no_han_review"),
            ("zh-Hant", "請你告訴我真正的名字", "no_han_review"),
        ]
        for language, caption, code in examples:
            with self.subTest(language=language, caption_length=len(caption)):
                result = audit_subtitles(
                    [C(1, 3, caption)],
                    options=AuditOptions(requested_language=language),
                )
                candidate = result["candidate"]
                self.assertNotIn(code, candidate["warning_counts"])
                self.assertNotIn("no_hangul_review", candidate["warning_counts"])
                self.assertTrue(candidate["script_warnings_are_heuristic_only"])
                self.assertFalse(result["accuracy_verified"])

    def test_foreign_language_is_flagged_per_requested_script_not_replaced(self):
        for lang, text, expected in (
            ("ko", "English subtitle", "no_hangul_review"),
            ("en", "한글 자막", "no_latin_review"),
            ("ja", "English only", "no_japanese_script_review"),
            ("zh", "English only", "no_han_review"),
            ("zh-Hans", "한국어", "no_han_review"),
            ("zh-Hant", "English only", "no_han_review"),
        ):
            with self.subTest(lang=lang):
                original = C(1, 3, text)
                candidate = audit_subtitles(
                    [original], options=AuditOptions(requested_language=lang),
                )["candidate"]
                self.assertEqual(candidate["warning_counts"].get(expected), 1)
                self.assertEqual(original.text, text)
                self.assertNotIn(text, json.dumps(candidate, ensure_ascii=False))

    def test_mixed_language_dialogue_is_not_wrongly_rejected(self):
        mixed = C(2, 5, "영어 I'm okay, 한국어 자막")
        result = audit_subtitles(
            [mixed], options=AuditOptions(requested_language="ko"),
        )
        self.assertNotIn("no_hangul_review", result["candidate"]["warning_counts"])
        self.assertEqual(mixed.text, "영어 I'm okay, 한국어 자막")

    def test_persistent_watermark_only_is_reviewed_without_cue_deletion(self):
        cue = C(0.125, 7.125, "방송 안내")
        report = audit_subtitles(
            [cue], options=AuditOptions(
                requested_language="ko", video_duration_seconds=7.125,
            ),
        )
        self.assertEqual(
            report["candidate"]["warning_counts"].get(
                "near_full_runtime_single_cue_review"
            ), 1,
        )
        self.assertEqual(report["candidate"]["cue_count"], 1)
        self.assertTrue(report["candidate"]["independent_video_duration_supplied"])
        self.assertFalse(report["accuracy_verified"])
        self.assertEqual(cue.text, "방송 안내")

    def test_genuine_full_length_single_cue_is_warning_not_auto_removed(self):
        cue = C(0.0, 7.125, "오늘 수업은 여기까지입니다")
        candidate = audit_subtitles(
            [cue], options=AuditOptions(video_duration_seconds=7.125)
        )["candidate"]
        self.assertIn("near_full_runtime_single_cue_review", candidate["warning_counts"])
        self.assertEqual(candidate["cue_count"], 1)
        self.assertTrue(candidate["warnings_are_not_verified_ocr_errors"])
        self.assertEqual(cue.text, "오늘 수업은 여기까지입니다")

    def test_no_source_duration_never_fabricates_complete_video_coverage(self):
        cue = C(0.0, 7.125, "뉴스 안내")
        candidate = audit_subtitles([cue])["candidate"]
        self.assertNotIn(
            "near_full_runtime_single_cue_review", candidate["warning_counts"]
        )
        self.assertFalse(candidate["independent_video_duration_supplied"])

    def test_dialogue_cues_in_same_video_are_not_full_length_single_cue(self):
        report = audit_subtitles(
            [C(1.5, 3.625, "첫 문장"), C(4.25, 6.375, "둘째 문장")],
            options=AuditOptions(video_duration_seconds=7.125),
        )
        self.assertNotIn(
            "near_full_runtime_single_cue_review",
            report["candidate"]["warning_counts"],
        )

    def test_proven_independent_short_dialogue_not_marked_persistent(self):
        cue = C(1.25, 4.0, "정상 대사")
        result = audit_subtitles(
            [cue], options=AuditOptions(video_duration_seconds=7.125),
        )["candidate"]
        self.assertNotIn(
            "near_full_runtime_single_cue_review", result["warning_counts"],
        )

    def test_cue_timeline_extending_beyond_known_video_is_review_not_deleted(self):
        cue = C(1.0, 8.25, "자막이 길어졌습니다")
        report = audit_subtitles(
            [cue], options=AuditOptions(video_duration_seconds=7.125),
        )
        self.assertEqual(
            report["candidate"]["warning_counts"].get(
                "cue_exceeds_video_duration_review"
            ), 1,
        )
        self.assertEqual(cue.end, 8.25)

    def test_invalid_duration_and_language_fail_before_report_creation(self):
        cases = [
            AuditOptions(video_duration_seconds=0),
            AuditOptions(video_duration_seconds=float("inf")),
            AuditOptions(video_duration_seconds=float("nan")),
            AuditOptions(video_duration_seconds=-1),
            AuditOptions(video_duration_seconds=21601),
            AuditOptions(video_duration_seconds=True),
            AuditOptions(video_duration_seconds="7.125"),
            AuditOptions(requested_language="unsupported"),
            AuditOptions(requested_language=None),
        ]
        for item in cases:
            with self.subTest(item=item), self.assertRaisesRegex(
                ValueError, "quality_audit_invalid_options"
            ):
                audit_subtitles([C(1, 2, "대사")], options=item)



if __name__ == "__main__":
    unittest.main()
