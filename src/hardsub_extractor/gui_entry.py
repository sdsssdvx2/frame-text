from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_fatal_json(dest: str | None, exc: BaseException) -> None:
    if not dest:
        return
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": False,
                "fatal": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_cli_selftest(argv: list[str]) -> int | None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-json")
    ap.add_argument("--self-test-progress")
    args, _ = ap.parse_known_args(argv)
    if not args.self_test and not args.self_test_json and not args.self_test_progress:
        return None

    try:
        from .selftest import run_selftest

        report = run_selftest(
            require_models=True,
            progress_path=args.self_test_progress,
        )
        payload = report.to_json()
        if args.self_test_json:
            dest = Path(args.self_test_json)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(payload, encoding="utf-8")
        return 0 if report.ok else 2
    except BaseException as exc:
        # Windowed PyInstaller applications may otherwise display a modal crash
        # dialog that hangs unattended CI. Self-test must always fail closed.
        _write_fatal_json(args.self_test_json, exc)
        if args.self_test_progress:
            progress = Path(args.self_test_progress)
            progress.parent.mkdir(parents=True, exist_ok=True)
            with progress.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "stage": "fatal",
                            "detail": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return 3



def _run_cli_browser_acceptance(argv: list[str]) -> int | None:
    """Run a bounded real browser-render check without opening a Tk window."""
    if "--browser-acceptance" not in argv:
        return None

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--browser-acceptance", action="store_true")
    ap.add_argument("--browser-acceptance-url", required=True)
    ap.add_argument("--browser-acceptance-json", required=True)
    ap.add_argument("--browser-acceptance-start", type=float, default=30.0)
    ap.add_argument("--browser-acceptance-seconds", type=float, default=30.0)
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3

    try:
        from .frozen_browser_acceptance import run_frozen_browser_acceptance

        return run_frozen_browser_acceptance(
            args.browser_acceptance_url,
            args.browser_acceptance_json,
            start_seconds=args.browser_acceptance_start,
            seconds=args.browser_acceptance_seconds,
        )
    except BaseException as exc:
        _write_fatal_json(args.browser_acceptance_json, exc)
        return 3


def _run_cli_caption_acceptance(argv: list[str]) -> int | None:
    """Use the bundled caption adapter in a headless frozen product test.

    Never open Tk, read browser credentials, download video, or preserve
    subtitle text. In contrast to a diagnostic probe, this acceptance gate
    requires a verified downloaded caption and exits nonzero otherwise.
    """
    if "--caption-acceptance" not in argv:
        return None

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--caption-acceptance", action="store_true")
    ap.add_argument("--caption-url", required=True)
    ap.add_argument("--caption-language", required=True)
    ap.add_argument("--caption-report", required=True)
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3

    try:
        from .runtime_paths import bundle_root
        from .public_caption_diagnostics import main as run_diagnostic

        downloader = bundle_root() / "bin" / "yt-dlp.exe"
        if not downloader.is_file():
            raise RuntimeError("bundled subtitle downloader unavailable")
        return run_diagnostic([
            "--url", args.caption_url,
            "--languages", args.caption_language,
            "--yt-dlp", str(downloader),
            "--max-seconds", "180",
            "--report", args.caption_report,
            "--require-download",
        ])
    except BaseException:
        # Absolute media URLs, yt-dlp stderr or profile paths must not be
        # written to CI diagnostics even when the frozen app fails.
        try:
            target = Path(args.caption_report)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({
                    "schema": 1, "type": "public_captions_frozen",
                    "status": "diagnostic_failed", "files_verified": False,
                }), encoding="utf-8",
            )
        except OSError:
            pass
        return 3



def _run_cli_multilingual_ocr_acceptance(argv: list[str]) -> int | None:
    """Exercise verified four-language OCR on local video inside frozen EXE.

    The command cannot fetch media or select arbitrary network sources.
    All synthesized test video inputs are local, bounded and verified by the
    acceptance module, and the resulting report excludes OCR/video contents.
    """
    if "--multilingual-ocr-acceptance" not in argv:
        return None
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--multilingual-ocr-acceptance", action="store_true")
    ap.add_argument("--ocr-fixture-dir", required=True)
    ap.add_argument("--ocr-acceptance-report", required=True)
    ap.add_argument(
        "--ocr-acceptance-backend",
        choices=("auto", "onnxruntime", "openvino"),
        default="onnxruntime",
    )
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3
    try:
        from .frozen_multilingual_ocr_acceptance import (
            run_frozen_multilingual_ocr_acceptance,
        )
        return run_frozen_multilingual_ocr_acceptance(
            args.ocr_fixture_dir, args.ocr_acceptance_report,
            inference_engine=args.ocr_acceptance_backend,
        )
    except BaseException as exc:
        try:
            destination = Path(args.ocr_acceptance_report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "schema": 1,
                "type": "frozen_multilingual_ocr_video_acceptance",
                "is_frozen": False,
                "ok": False,
                "failure_type": type(exc).__name__,
                "original_text_exact_total": 0,
            }), encoding="utf-8")
        except OSError:
            pass
        return 3


def _run_cli_local_embedded_acceptance(argv: list[str]) -> int | None:
    """Headless frozen EXE: synthetic local MKV embedded text, no user media."""
    if "--local-embedded-acceptance" not in argv:
        return None
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--local-embedded-acceptance", action="store_true")
    ap.add_argument("--local-embedded-report", required=True)
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3

    try:
        from .frozen_local_embedded_acceptance import (
            run_frozen_local_embedded_acceptance,
        )
        return run_frozen_local_embedded_acceptance(args.local_embedded_report)
    except BaseException as exc:
        # Never overwrite a report or leak an exception path, caption text,
        # media URL, full command line, browser content, or secrets.
        try:
            target = Path(args.local_embedded_report)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as output:
                json.dump({
                    "schema": 1,
                    "type": "frozen_local_embedded_video_acceptance",
                    "is_frozen": False,
                    "ok": False,
                    "failure_type": type(exc).__name__,
                    "exact_cue_count": 0,
                }, output)
        except OSError:
            pass
        return 3


def _run_cli_frozen_public_video_ocr(argv: list[str]) -> int | None:
    """Frozen-only isolated real video OCR with fake public media transport."""
    if "--public-video-ocr-acceptance" not in argv:
        return None
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--public-video-ocr-acceptance", action="store_true")
    ap.add_argument("--ocr-fixture-dir", required=True)
    ap.add_argument("--public-ocr-report", required=True)
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3
    try:
        from .frozen_public_video_ocr_acceptance import (
            run_frozen_public_video_ocr_acceptance,
        )
        return run_frozen_public_video_ocr_acceptance(
            args.ocr_fixture_dir, args.public_ocr_report,
        )
    except BaseException as exc:
        try:
            destination = Path(args.public_ocr_report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as output:
                json.dump({
                    "schema": 1,
                    "type": "frozen_public_video_ocr_fallback_acceptance",
                    "is_frozen": False,
                    "ok": False,
                    "failure_type": type(exc).__name__,
                    "exact_total": 0,
                }, output)
        except OSError:
            pass
        return 3


def _run_cli_frozen_frame_review(argv: list[str]) -> int | None:
    """Probe real frozen GUI review worker without displaying Tk."""
    if "--frozen-frame-review-acceptance" not in argv:
        return None
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--frozen-frame-review-acceptance", action="store_true")
    ap.add_argument("--review-acceptance-report", required=True)
    try:
        args, _ = ap.parse_known_args(argv)
    except SystemExit:
        return 3
    try:
        from .frozen_frame_review_acceptance import (
            run_frozen_frame_review_acceptance,
        )
        return run_frozen_frame_review_acceptance(args.review_acceptance_report)
    except BaseException:
        return 3


def main() -> None:
    import sys

    rc = _run_cli_selftest(sys.argv[1:])
    if rc is None:
        rc = _run_cli_browser_acceptance(sys.argv[1:])
    if rc is None:
        rc = _run_cli_caption_acceptance(sys.argv[1:])
    if rc is None:
        rc = _run_cli_multilingual_ocr_acceptance(sys.argv[1:])
    if rc is None:
        rc = _run_cli_local_embedded_acceptance(sys.argv[1:])
    if rc is None:
        rc = _run_cli_frozen_public_video_ocr(sys.argv[1:])
    if rc is None:
        rc = _run_cli_frozen_frame_review(sys.argv[1:])
    if rc is not None:
        raise SystemExit(rc)

    # Keep GUI/current-player imports lazy so CI can exercise the frozen
    # runtime self-test without creating a Tk window or initializing network
    # clients. The GUI itself keeps importing the legacy AutoEpSiteClient for
    # compatibility, but the Windows entry point replaces that one dependency
    # with the behavior-compatible current-player transport cascade.
    from . import gui
    from .site_adapter import SITE_ADAPTER

    if SITE_ADAPTER is not None:  # private build only
        gui.AutoEpSiteClient = SITE_ADAPTER.CurrentAutoEpSiteClient
    gui.main()


if __name__ == "__main__":
    main()
