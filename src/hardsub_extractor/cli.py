import argparse
from pathlib import Path
import urllib.parse

from .site_adapter import SITE_ADAPTER, is_site_episode_url
from .diagnostics import DiagnosticRecorder
from .frame_source import NormalizedROI
from .ocr.adaptive import AdaptiveOCREngine
from .ocr.rapidocr_adapter import RapidOCRAdapter
from .model_assets import normalize_ocr_language
from .runtime_paths import bundled_model_dir
from .pipeline import extract
from .extraction_progress import ExtractionMonitor
from .local_soft_subtitles import acquire_local_embedded_text
from .local_image_subtitles import acquire_local_embedded_image
from .local_sidecar_subtitles import acquire_local_sidecar_text


# Optional private site adapter (absent in the public build).
extract_epsite = getattr(SITE_ADAPTER, "extract_epsite", None)
extract_and_sync_epsite = getattr(SITE_ADAPTER, "extract_and_sync_epsite", None)


def _is_epsite_url(value: str) -> bool:
    return is_site_episode_url(value)


def _ocr(args):
    language = normalize_ocr_language(getattr(args, "ocr_language", "ko"))
    if args.opencv_recognizer:
        if language != "ko":
            raise SystemExit("OpenCV fallback is not verified for non-Korean OCR")
        if not args.recognizer_model or not args.recognizer_dict:
            raise SystemExit(
                "--opencv-recognizer requires --recognizer-model and --recognizer-dict"
            )
        from .ocr.opencv_recognizer import OpenCVSubtitleRecognizer
        base = OpenCVSubtitleRecognizer(
            args.recognizer_model,
            args.recognizer_dict,
        )
    else:
        model_dir = args.model_dir
        if language != "ko" and not model_dir:
            model_dir = str(bundled_model_dir())
        base = RapidOCRAdapter(
            text_score=args.text_score,
            model_dir=model_dir,
            language=language,
            inference_engine=getattr(args, "ocr_backend", "onnxruntime"),
        )
    selected = AdaptiveOCREngine(base)
    if getattr(args, "caption_islands", False):
        # Optional, not default: filtering may invoke existing bounded adaptive retries.
        from .ocr.caption_islands import CaptionIslandEngine
        return CaptionIslandEngine(selected)
    if getattr(args, "light_caption_only", False):
        # Explicit bright-on-dark style requirement, not a language guess.
        # Incompatible with dark/colored captions; never silently enable.
        from .ocr.light_text import LightCaptionEngine
        selected = LightCaptionEngine(selected)
        if getattr(args, "light_caption_temporal", False):
            from .ocr.light_temporal import LightTemporalWitnessEngine
            selected = LightTemporalWitnessEngine(selected)
        return selected
    return selected


def _diag(args):
    return DiagnosticRecorder(args.diagnostics) if args.diagnostics else None


def main():
    p=argparse.ArgumentParser(prog="hard-sub-extract")
    p.add_argument("source",help="local video path (private builds: site episode URL)")
    p.add_argument("--out",default="extracted-subtitles")
    p.add_argument("--epsite",action="store_true")
    p.add_argument("--target-video",default=None,
                   help="HQ target video; with EpSite, extract and visual-sync in one run")
    p.add_argument("--model-dir",default=None)
    p.add_argument("--ocr-language",choices=("ko", "en", "ja", "zh"),default="ko",
                   help="Original subtitle language for embedded text first, then burned-in OCR fallback")
    p.add_argument("--ocr-backend", choices=("auto", "onnxruntime", "openvino"),
                   default="auto",
                   help="OCR CPU backend. auto prefers bundled OpenVINO and falls back to ONNX Runtime; explicit choices remain available.")
    p.add_argument("--subtitle-file", default=None,
                   help="Explicit original local SRT/VTT/ASS/SSA file; takes priority over embedded tracks and OCR")
    p.add_argument("--hard-only",action="store_true",
                   help="Local video only: skip embedded text and run hard-subtitle OCR for comparison")
    p.add_argument("--focus-bottom-two-lines", action="store_true",
                   help="EXPERIMENTAL local RapidOCR only: when 3+ visual lines are recognized, keep the last two; may discard genuine dialogue; default OFF")
    p.add_argument("--caption-islands", action="store_true",
                   help="EXPERIMENTAL: coherent text blocks instead of accepting disconnected OCR rows (default OFF)")
    p.add_argument("--light-caption-only", action="store_true",
                   help="EXPERIMENTAL: require light glyphs on darker backgrounds; discards dark/colored subtitles; default OFF")
    p.add_argument("--light-caption-temporal", action="store_true",
                   help="EXPERIMENTAL local video only: infer timing from existing OCR glyph masks and source frames; no additional model calls")
    p.add_argument("--native-vsf-executable", default=None,
                   help="Optional local official GPL-2.0 VideoSubFinder 6.10.5 CLI. Never downloaded or bundled by HSE; original subtitle tracks still take precedence.")
    p.add_argument("--native-vsf-sha256", default=None,
                   help="Independently verified SHA-256 of the explicitly selected official VideoSubFinder executable.")
    p.add_argument("--seconv-official-54", action="store_true",
                   help="Explicitly opt in to SHA-pinned free SeConv 5.4 for matched PGS/VobSub bitmap tracks; bundled SeConv 5.3 stays default.")
    p.add_argument("--native-vsf-official", action="store_true",
                   help="Explicit opt-in: fetch SHA-pinned free official VideoSubFinder into private user cache only if local soft/image subtitle tracks are absent.")
    p.add_argument("--native-vsf-split-stable", action="store_true",
                   help="EXPERIMENTAL: split two high-confidence consecutive OCR sentences from one native timing interval (default OFF).")
    p.add_argument("--native-vsf-official-version", choices=("v6.10.5", "v6.10.7"), default="v6.10.5",
                   help="Choose separately pinned official stable VSF version (newer is research-only, default preserves 6.10.5).")
    p.add_argument("--opencv-recognizer",action="store_true",
                   help="use OpenCV DNN recognizer-only fallback instead of RapidOCR")
    p.add_argument("--recognizer-model",default=None)
    p.add_argument("--recognizer-dict",default=None)
    p.add_argument("--diagnostics",default=None)
    p.add_argument("--roi-x",type=float,default=.04)
    p.add_argument("--roi-y",type=float,default=.58)
    p.add_argument("--roi-w",type=float,default=.92)
    p.add_argument("--roi-h",type=float,default=.40)
    p.add_argument("--text-score",type=float,default=.45)
    p.add_argument("--sync-max-offset",type=float,default=20.0)
    p.add_argument("--sync-window",type=float,default=8.0)
    p.add_argument("--sync-anchor-spacing",type=float,default=4.0)
    p.add_argument("--sync-sample-fps",type=float,default=4.0)
    a=p.parse_args()
    if a.caption_islands and a.light_caption_only:
        p.error("select only one experimental subtitle-style selector")
    # The optional spatial filter needs RapidOCR geometry from a local
    # video: do not silently enable it on unverified transports/engines.
    if a.caption_islands and (
        a.epsite or _is_epsite_url(a.source)
        or a.native_vsf_executable or a.native_vsf_official
        or a.opencv_recognizer
    ):
        p.error("--caption-islands requires local RapidOCR without native VSF")
    if a.light_caption_only and a.opencv_recognizer:
        p.error("light-caption-only needs RapidOCR text-box geometry")
    if a.opencv_recognizer and a.ocr_backend == "openvino":
        p.error("--ocr-backend=openvino applies only to RapidOCR")
    if a.light_caption_temporal and not a.light_caption_only:
        p.error("--light-caption-temporal requires --light-caption-only")
    # Browser-rendered fallback may use a single-pass, bounded in-memory
    # glyph journal. A resolved remote HLS/HTTP input is still unsupported.
    # The EpSite adapter checks that the actual source is browser-rendered.
    if a.light_caption_temporal and (a.native_vsf_executable or a.native_vsf_official):
        p.error("--light-caption-temporal is not connected to native VSF extraction")
    if a.light_caption_temporal and not (a.epsite or _is_epsite_url(a.source)):
        # Local source is mandatory for the FFmpeg second pass.
        local_source = Path(a.source).expanduser()
        if (local_source.is_symlink() or not local_source.is_file()
                or local_source.stat().st_size <= 0):
            p.error("--light-caption-temporal requires a regular local video file")

    roi=NormalizedROI(a.roi_x,a.roi_y,a.roi_w,a.roi_h)
    is_epsite=a.epsite or _is_epsite_url(a.source)
    if a.focus_bottom_two_lines and (
        is_epsite or a.native_vsf_executable or a.native_vsf_official
        or a.caption_islands or a.light_caption_only or a.opencv_recognizer
    ):
        p.error("--focus-bottom-two-lines requires standard local visual RapidOCR (no native VSF or other OCR style)")
    if a.subtitle_file and (is_epsite or a.hard_only):
        p.error("--subtitle-file requires local video and cannot combine with --hard-only")
    if bool(a.native_vsf_executable) != bool(a.native_vsf_sha256):
        p.error("--native-vsf-executable and --native-vsf-sha256 must be supplied together")
    if (a.native_vsf_executable or a.native_vsf_official) and is_epsite:
        p.error("native VSF is only allowed for local videos")
    if a.native_vsf_official and (a.native_vsf_executable or a.native_vsf_sha256):
        p.error("official fetch opt-in cannot combine with manual native executable")
    if a.native_vsf_official_version != "v6.10.5" and not a.native_vsf_official:
        p.error("selecting a newer official VSF version requires --native-vsf-official")
    if a.native_vsf_split_stable and not (
        a.native_vsf_executable or a.native_vsf_official
    ):
        p.error("stable subtitle split requires explicit native VSF selection")
    if a.seconv_official_54 and (is_epsite or a.hard_only):
        p.error("official SeConv 5.4 selection requires original-track-first local video")
    diagnostics=_diag(a)

    if a.light_caption_temporal and is_epsite and a.target_video:
        p.error("--light-caption-temporal cannot combine with EpSite target-video sync")
    if is_epsite and SITE_ADAPTER is None:
        p.error("site episode adapter is not included in this build; use a local video or public caption URL")
    if is_epsite:
        # Private build only (guarded above).
        from .sources.epsite_current_client import CurrentAutoEpSiteClient
        client=CurrentAutoEpSiteClient()

        # Keep resolution inside extract_epsite so every entry point shares the
        # exact same soft -> public-HLS hard OCR -> browser-render hard OCR
        # decision. OCR remains lazy and is never loaded for a valid soft track.
        ocr_factory=lambda: _ocr(a)

        if a.target_video:
            r=extract_and_sync_epsite(
                a.source,a.target_video,Path(a.out),
                ocr=None,ocr_factory=ocr_factory,ocr_language=a.ocr_language,
                client=client,diagnostics=diagnostics,
                subtitle_roi=roi,
                sync_sample_fps=a.sync_sample_fps,
                sync_window_seconds=a.sync_window,
                sync_anchor_spacing=a.sync_anchor_spacing,
                sync_max_abs_offset=a.sync_max_offset,
                streaming=True,
            )
            print(f"mode={r.extracted.mode}+visual-piecewise-sync")
            print(f"source-cues={r.extracted.cue_count} final-cues={len(r.synced.cues)}")
            print(r.synced.report_path)
            if diagnostics:
                print(diagnostics.write(extra={
                    "mode":r.extracted.mode+"+visual-piecewise-sync",
                    "source":a.source,
                    "target_video":a.target_video,
                }))
            print(r.synced.srt_path);print(r.synced.ass_path)
            return

        r=extract_epsite(
            a.source,Path(a.out),client=client,roi=roi,
            diagnostics=diagnostics,ocr_factory=ocr_factory,
            ocr_language=a.ocr_language,
            streaming=True, browser_temporal=a.light_caption_temporal,
        )
        print(f"mode={r.mode} cues={r.cue_count}")
        if r.source_subtitle_url:
            print(f"soft-subtitle={r.source_subtitle_url}")
        if diagnostics:
            print(diagnostics.write(extra={"mode":r.mode,"source":a.source}))
        print(r.srt_path);print(r.ass_path)
        return

    if not a.hard_only:
        if a.subtitle_file:
            # Explicit selection cannot silently fall back to a different
            # embedded track or OCR when the selected file is invalid.
            soft = acquire_local_sidecar_text(
                a.source, Path(a.out), language=a.ocr_language,
                selected_path=a.subtitle_file,
            )
        else:
            soft = acquire_local_embedded_text(
                a.source, Path(a.out), language=a.ocr_language,
            )
            if soft is None:
                soft = acquire_local_sidecar_text(
                    a.source, Path(a.out), language=a.ocr_language,
                )
            if soft is None:
                converter_factory = None
                if a.seconv_official_54:
                    from .managed_seconv import ensure_official_seconv_54
                    converter_factory = lambda: ensure_official_seconv_54(consent=True)
                soft = acquire_local_embedded_image(
                    a.source, Path(a.out), language=a.ocr_language,
                    ocr_factory=lambda: _ocr(a),
                    **({"converter_factory": converter_factory}
                       if converter_factory is not None else {}),
                )
        if soft is not None:
            print(f"mode={soft.mode} cues={soft.cue_count}")
            print(soft.provenance_path)
            print(soft.srt_path)
            print(soft.ass_path)
            return

    if a.native_vsf_executable or a.native_vsf_official:
        # Explicit opt-in ONLY, and only after the source-first path above
        # has returned no matched original subtitle. The official archive is
        # pinned and cached separately; never bundled or system-installed.
        from .native_vsf_timing import extract_with_native_vsf
        executable, digest = a.native_vsf_executable, a.native_vsf_sha256
        if a.native_vsf_official:
            # Do not download even the trusted optional native tool when the
            # supplied media is not an existing LOCAL file or the destination
            # would collide with already-present subtitle outputs.
            video = Path(a.source).expanduser()
            if (
                video.is_symlink() or not video.is_file()
                or video.suffix.lower() not in {
                    ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts", ".avi",
                }
                or video.stat().st_size <= 0
            ):
                p.error("optional official native timing requires a regular local video")
            base = Path(a.out).expanduser()
            if any(
                dest.exists() or dest.is_symlink()
                for dest in (base.with_suffix(".srt"), base.with_suffix(".ass"))
            ):
                p.error("existing subtitle outputs must be preserved")
            from .native_vsf_assets import ensure_official_vsf
            executable, digest = ensure_official_vsf(
                consent=True,
                **({"release_version": a.native_vsf_official_version}
                   if a.native_vsf_official_version != "v6.10.5" else {}),
            )
        finished = extract_with_native_vsf(
            a.source, Path(a.out), executable=executable,
            pinned_executable_sha256=digest,
            ocr_factory=lambda: _ocr(a), roi=roi,
            **({"split_stable_text": True}
               if a.native_vsf_split_stable else {}),
        )
        report = finished.report
        print(
            f"mode=hard-ocr-native-timing "
            f"intervals={report.detected_intervals} "
            f"missed={report.missed_intervals} "
            f"cues={getattr(report, 'exported_cues', report.recognized_intervals)} "
            f"split_intervals={getattr(report, 'split_source_intervals', 0)} "
            f"ocr_calls={report.ocr_calls}"
        )
        short = getattr(report, "short_frame_rescue_attempted", 0)
        deferred = getattr(report, "short_frame_rescue_deferred", 0)
        errors = getattr(report, "short_frame_rescue_errors", 0)
        if short or deferred or errors:
            print(
                "short_frame_rescue_attempted=" + str(short)
                + " recovered=" + str(getattr(
                    report, "short_frame_rescue_recognized", 0
                ))
                + " deferred=" + str(deferred)
                + " errors=" + str(errors)
            )
        split_cues = tuple(getattr(report, "split_cue_numbers", ()))[:8]
        if split_cues:
            print("split_pair_first_cue_numbers_first_8=" +
                  ",".join(str(int(n)) for n in split_cues))
        windows = tuple(getattr(report, "missed_windows", ()))
        if windows:
            from .timed_ocr import format_missed_windows
            print("ocr_unreadable_candidate_times_first_8=" +
                  format_missed_windows(windows))
        print(report.srt_path);print(report.ass_path)
        return

    # The direct local CLI should reject an occupied destination before
    # loading OCR models or scanning a full-length video.
    from .writers import ensure_subtitle_pair_available
    ensure_subtitle_pair_available(a.out)
    selected_ocr = _ocr(a)
    report=extract(
        a.source,selected_ocr,Path(a.out),roi=roi,diagnostics=diagnostics,
        monitor=ExtractionMonitor(exclusive_outputs=True),
        streaming=True,
        timing_refiner=(
            selected_ocr.refine_timings if a.light_caption_temporal else None
        ),
        **({"focus_bottom_two_lines": True} if a.focus_bottom_two_lines else {}),
    )
    print(f"mode=hard-ocr segments={report.segment_count} cues={report.cue_count}")
    if diagnostics:
        print(diagnostics.write(extra={"mode":"hard-ocr","source":a.source}))
    print(report.srt_path);print(report.ass_path)

if __name__=="__main__":
    main()
