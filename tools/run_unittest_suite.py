from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path


LOOPBACK_CASES = {
    "EpSiteDirectTests": {
        "test_auto_client_falls_back_to_curl_with_session",
        "test_auto_client_tries_urllib_if_curl_resume_transport_fails",
        "test_auto_client_uses_chromium_page_after_both_http_entry_fetches_fail",
        "test_direct_hardsub_resolution",
        "test_direct_http_status_is_not_browser_transport",
        "test_direct_media_playlist_is_accepted",
        "test_ffmpeg_args_include_user_agent_and_referer",
        "test_soft_subtitle_end_to_end_normalization",
        "test_soft_subtitle_has_priority",
    },
    "HLSHeaderTests": {
        "test_probe_and_decode_both_forward_headers",
    },
    # This real Playwright/Chromium test requires an unmodified localhost
    # HTTP route, and consistently times out on the contaminated Windows
    # runner. Skip only after a failed preflight, never on a clean host.
    "BrowserEncodedWebMTests": {
        "test_actual_chromium_encodes_and_streams_tab_pixels_through_real_page",
    },
    # On a preflight-confirmed corrupt localhost route, HTTP finish ACKs
    # can time out after an unrelated intermediary alters the exchange.
    # Keep the exact clean-host acceptance test and test the session-owned
    # offset invariant independently of HTTP below.
    "TabCaptureBridgeTests": {
        "test_original_start_offset_is_pinned_on_finish_retry",
    },
    # Actual Chromium localhost navigation also fails with ERR_CONNECTION_RESET
    # on this host. It must run on a clean host, not be treated as passed.
    "TabStopAcceptance": {
        "test_host_stop_closes_actual_browser_media_track_and_saves_once",
    },
    # The same already-failed loopback preflight also makes this real
    # Chromium MediaRecorder -> localhost chunk/finish test unreliable.
    # Both modes remain mandatory on clean hosts. Explicitly count them
    # as NOT VERIFIED, never as successful browser product acceptance.
    "AutoStopChromeIntegration": {
        "test_15_second_recorder_stop_ocr_and_gui_server_release",
        "test_explicit_persistent_white_chromium_recording_and_export",
    },
}

DETERMINISTIC_CASES = {
    "EpSiteLegacyDeterministicTests": {
        "test_auto_client_uses_scripted_secondary_after_primary_transport_failure",
        "test_browser_page_resume_prefers_secondary_transport",
        "test_browser_page_resume_tries_primary_after_secondary_transport_failure",
        "test_cookie_jar_applies_same_origin_player_session_before_network",
        "test_direct_hardsub_resolution_and_getvideo_contract",
        "test_direct_http_error_remains_hard_error_not_browser_transport",
        "test_direct_media_playlist_is_accepted_without_variant_guessing",
        "test_soft_subtitle_end_to_end_normalization",
        "test_soft_subtitle_priority_and_download",
    },
    "HLSHeaderCommandTests": {
        "test_ffprobe_places_user_agent_and_referer_before_input_url",
        "test_ffmpeg_places_headers_and_seek_before_input_duration_after",
    },
    # Offline local browser capture integrity remains mandatory, including
    # real local HTTP chunk receipt and preservation of existing outputs.
    "TabCaptureBridgeTests": {
        "test_direct_finish_offset_commits_once_without_http_transport",
        "test_cross_origin_host_and_missing_token_blocked",
        "test_chunk_sequence_bounds_and_no_overwrite",
        "test_finish_calls_local_extractor_once_and_preserves_originals",
    },
}

SKIP_REASON = (
    "self-hosted Windows loopback integrity preflight failed: localhost HTTP "
    "headers/connections are being modified outside HSE; the real browser "
    "capture and finish-retry HTTP tests remain separately unverified; "
    "deterministic local transport/offset contract regressions remain mandatory"
)


def iter_cases(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_cases(item)
        else:
            yield item


def _expected_case_pairs(mapping: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {
        (class_name, method_name)
        for class_name, methods in mapping.items()
        for method_name in methods
    }


def validate_deterministic_contract(suite: unittest.TestSuite) -> list[str]:
    expected = _expected_case_pairs(DETERMINISTIC_CASES)
    selected: list[str] = []
    observed: set[tuple[str, str]] = set()
    for case in iter_cases(suite):
        pair = (case.__class__.__name__, getattr(case, "_testMethodName", ""))
        if pair in expected:
            observed.add(pair)
            selected.append(case.id())

    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "deterministic loopback replacement contract drifted; "
            f"expected={len(expected)} selected={len(selected)} "
            f"missing={missing} unexpected={unexpected}"
        )
    return sorted(selected)


def apply_environment_skips(suite: unittest.TestSuite) -> list[str]:
    if os.environ.get("HSE_SKIP_LOOPBACK_INTEGRATION") != "1":
        return []

    expected = _expected_case_pairs(LOOPBACK_CASES)
    selected: list[str] = []
    observed: set[tuple[str, str]] = set()
    for case in iter_cases(suite):
        pair = (case.__class__.__name__, getattr(case, "_testMethodName", ""))
        if pair not in expected:
            continue
        observed.add(pair)
        method_name = case._testMethodName
        original = getattr(case.__class__, method_name)
        if not getattr(original, "__unittest_skip__", False):
            setattr(case.__class__, method_name, unittest.skip(SKIP_REASON)(original))
        selected.append(case.id())

    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "loopback environment skip contract drifted; "
            f"expected={len(expected)} selected={len(selected)} "
            f"missing={missing} unexpected={unexpected}"
        )
    return sorted(selected)


def write_report(
    path: str | None,
    *,
    requested: list[str],
    observed: list[str],
    deterministic: list[str],
) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": 3,
                "loopback_environment_skip_enabled": bool(requested),
                "requested_skip_count": len(requested),
                "observed_skip_count": len(observed),
                "requested_test_ids": requested,
                "observed_test_ids": observed,
                "reason": SKIP_REASON if requested else None,
                "deterministic_replacement_count": len(deterministic),
                "deterministic_replacement_test_ids": deterministic,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    # Running `python tools/run_unittest_suite.py` places tools/, not the
    # repository root, on sys.path. Keep the root importable for the
    # historical tools.* namespace-package compatibility shims.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    suite = unittest.defaultTestLoader.discover("tests")
    deterministic = validate_deterministic_contract(suite)
    requested = apply_environment_skips(suite)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    observed = sorted(
        test.id()
        for test, reason in result.skipped
        if reason == SKIP_REASON
    )
    write_report(
        os.environ.get("HSE_TEST_SKIP_REPORT"),
        requested=requested,
        observed=observed,
        deterministic=deterministic,
    )
    if requested and observed != requested:
        print(
            "environment skip accounting mismatch: "
            f"requested={len(requested)} observed={len(observed)}",
            file=sys.stderr,
        )
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
