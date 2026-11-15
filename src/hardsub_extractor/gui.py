from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import os
import re
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .extraction_progress import ExtractionCancelled, ExtractionMonitor, ProgressEvent
from .site_adapter import SITE_ADAPTER, source_subtitle_base
from .diagnostics import DiagnosticRecorder
from .ocr.adaptive import AdaptiveOCREngine
from .ocr.rapidocr_adapter import RapidOCRAdapter, openvino_runtime_available
from .runtime_paths import (
    bundled_manifest_path,
    bundled_model_dir,
    resolve_binary,
)
from .selftest import run_selftest
from .pipeline import extract as extract_local_video
from .subtitle_batch import load_episode_urls
from .local_batch import load_local_batch, run_local_batch
from .subtitle_acquisition import CaptionAccessError, PublicCaptionCollector
from .subtitle_language import normalize_language
from .model_assets import normalize_ocr_language
from .local_soft_subtitles import acquire_local_embedded_text
from .local_image_subtitles import acquire_local_embedded_image
from .local_sidecar_subtitles import acquire_local_sidecar_text, _outputs as _all_local_outputs
from .subtitle_proxy import CaptionProxyRoute
from .public_video_ocr import PublicVideoOCRError, resolve_missing_public_ocr
from .managed_caption_downloader import (
    auto_update_enabled, set_auto_update_enabled,
    refresh as refresh_managed_downloader,
)

# Optional private site adapter (absent in the public build).
extract_epsite = getattr(SITE_ADAPTER, "extract_epsite", None)
extract_and_sync_epsite = getattr(SITE_ADAPTER, "extract_and_sync_epsite", None)
AutoEpSiteClient = getattr(SITE_ADAPTER, "AutoEpSiteClient", None)


_LOCAL_OCR_REGION_LABELS = {
    "기본 · 하단 40%": "bottom40",
    "좁게 · 하단 25% (윗줄 유실 가능)": "bottom25",
    "넓게 · 하단 65% (오검출 주의)": "lower65",
    "넓게 · 하단 80% (오검출 주의)": "lower80",
}


def _local_region_from_label(label: str) -> str:
    if type(label) is not str or label not in _LOCAL_OCR_REGION_LABELS:
        raise ValueError("invalid GUI OCR scope")
    return _LOCAL_OCR_REGION_LABELS[label]


_GUI_SECRET_HEADER_RE = re.compile(
    r"(?im)\b(?:authorization|proxy-authorization|cookie|x-api-key)\s*:\s*[^\r\n]+"
)
_GUI_URL_RE = re.compile(r"(?i)\b(?:https?|wss?|socks5h?)://[^\s\"'<>]+")


def _safe_gui_error_text(value: str) -> str:
    """Keep useful exception context without exposing player URLs or headers."""
    return _GUI_URL_RE.sub("<url>", _GUI_SECRET_HEADER_RE.sub("<redacted-header>", value))


def _user_error_summary(exc: Exception, *, limit: int = 900) -> str:
    text = _safe_gui_error_text(str(exc)).strip() or type(exc).__name__
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _open_owner_review_page(output: str) -> bool:
    """Open only this approved job's bounded local HTML; never a URL."""
    if os.name != "nt":
        return False
    folder = Path(output)
    page = folder / "review.html"
    if (folder.is_symlink() or page.is_symlink() or not page.is_file()
            or not 0 < page.stat().st_size <= 128 * 1024):
        return False
    os.startfile(str(page))
    return True


def _planned_subtitle_outputs(out: Path, *, target_sync: bool) -> tuple[Path, ...]:
    """Enumerate all output files a GUI extraction may overwrite.

    The HQ sync path also creates a separate source-timeline subtitle pair.
    Never mistake the two timelines for the same output; reserve both.
    """
    results = [out.with_suffix(".srt"), out.with_suffix(".ass")]
    if target_sync:
        source = source_subtitle_base(out)
        results.extend((
            source.with_suffix(".srt"),
            source.with_suffix(".ass"),
            out.with_suffix(".sync.json"),
        ))
    return tuple(results)


def _check_no_existing_subtitles(out: Path, *, target_sync: bool) -> None:
    for path in _planned_subtitle_outputs(out, target_sync=target_sync):
        # Path.exists() alone misses dangling symlinks on Windows/Linux.
        if path.exists() or path.is_symlink():
            raise ValueError(
                "이미 같은 이름의 자막/싱크 결과 파일이 있습니다. "
                "기존 파일을 보존하려면 다른 결과 파일 경로를 지정하세요."
            )


class ExtractorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hard Subtitle Extractor")
        self.root.geometry("820x760")
        self.root.minsize(720, 660)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._progress_stage: str | None = None
        self._pending_close = False
        self._tab_capture_thread = None
        self._tab_capture_stop = threading.Event()

        self.source_mode_var = tk.StringVar(
            value="epsite" if SITE_ADAPTER is not None else "local"
        )
        self.url_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.sidecar_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(Path.home() / "Desktop" / "extracted-subtitles"))
        self.diag_var = tk.BooleanVar(value=False)
        self.languages_var = tk.StringVar(value="ko")
        self.ocr_language_var = tk.StringVar(value="ko")
        default_fast_cpu = openvino_runtime_available()
        self.fast_cpu_var = tk.BooleanVar(value=default_fast_cpu)
        self.video_ocr_var = tk.BooleanVar(value=True)
        # Never enable this experimental GPL-2.0 external detector silently.
        self.native_vsf_var = tk.BooleanVar(value=False)
        self.native_split_var = tk.BooleanVar(value=False)
        self._active_native_split_opt_in = False
        self.focus_bottom_two_var = tk.BooleanVar(value=False)
        self._active_focus_bottom_two = False
        self.caption_islands_var = tk.BooleanVar(value=False)
        self._active_caption_islands = False
        self.light_caption_temporal_var = tk.BooleanVar(value=False)
        self._active_light_caption_temporal = False
        self.pixel_timing_only_var = tk.BooleanVar(value=False)
        self._active_pixel_timing_only = False
        self.seconv54_var = tk.BooleanVar(value=False)
        self._active_seconv54_opt_in = False
        self.native_vsf_release_var = tk.StringVar(value="v6.10.5")
        self._active_native_vsf_release = "v6.10.5"
        self._managed_caption_auto_on_start = auto_update_enabled()
        self.managed_caption_updates_var = tk.BooleanVar(
            value=self._managed_caption_auto_on_start,
        )
        self._managed_caption_update_running = False
        self._active_native_vsf_opt_in = False
        self._requested_video_ocr_enabled = True
        self._requested_ocr_language = "ko"
        self._requested_ocr_backend = (
            "auto" if default_fast_cpu else "onnxruntime"
        )
        self._active_ocr_backend = self._requested_ocr_backend
        self.proxy_mode_var = tk.StringVar(value="direct")
        self.proxy_url_var = tk.StringVar(value="")
        self._caption_proxy_route = CaptionProxyRoute()
        self._requested_languages: tuple[str, ...] = ("ko",)
        # Private, explicitly selected local batch inputs. GUI displays
        # only a count; actual paths go to the worker snapshot, never logs.
        self._local_batch_selection: tuple[str, ...] = ()
        self._local_batch_display = ""
        self._requested_local_batch_paths: tuple[str, ...] = ()
        self.status_var = tk.StringVar(value="준비됨")
        # Explicit opt-in local review, not a post-extraction automatic step.
        self.review_mode_var = tk.StringVar(value="경고 자막만")
        self.review_after_var = tk.StringVar(value="0")
        self.preview_time_var = tk.StringVar(value="00:01:00")
        self.local_ocr_region_var = tk.StringVar(value="기본 · 하단 40%")
        # Candidate cues are private, bounded and RAM-only. They are never
        # automatic deletion instructions or written to the user's subtitles.
        self._temporal_review_hints = None
        self._native_split_review_hints = None
        self._active_temporal_review_opt_in = False

        self._build()
        self._refresh_runtime_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)
        if self._managed_caption_auto_on_start:
            self._start_managed_caption_update(auto=True)

    def _make_scrollable_container(self):
        """Host the whole form in a vertically scrollable canvas.

        The form (main inputs + optional tools + log) is taller than common
        laptop screens, which previously pushed the start button and log
        below the visible window. Scrolling keeps every control reachable.
        """
        canvas = tk.Canvas(self.root, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_width(event):
            canvas.itemconfigure(window, width=event.width)

        def _sync_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", _sync_width)
        inner.bind("<Configure>", _sync_region)

        def _wheel(event):
            widget = event.widget
            # The log keeps its own scrolling; everything else scrolls the form.
            if isinstance(widget, tk.Text):
                return None
            try:
                if not str(widget).startswith(str(canvas)):
                    return None
            except Exception:
                return None
            if getattr(event, "num", None) == 4:
                step = -1
            elif getattr(event, "num", None) == 5:
                step = 1
            else:
                delta = int(getattr(event, "delta", 0) or 0)
                if delta == 0:
                    return None
                step = -1 if delta > 0 else 1
            canvas.yview_scroll(step * 3, "units")
            return "break"

        self.root.bind_all("<MouseWheel>", _wheel, add="+")
        self.root.bind_all("<Button-4>", _wheel, add="+")
        self.root.bind_all("<Button-5>", _wheel, add="+")
        self._scroll_canvas = canvas
        return inner

    def _fit_window_to_screen(self, outer):
        try:
            self.root.update_idletasks()
            wanted = int(outer.winfo_reqheight()) + 8
            screen = int(self.root.winfo_screenheight())
            height = max(600, min(wanted, int(screen * 0.88)))
            self.root.geometry(f"860x{height}")
        except Exception:
            pass

    def _toggle_advanced_tools(self):
        box = getattr(self, "_advanced_box", None)
        if box is None:
            return
        if self.show_advanced_var.get():
            box.grid()
        else:
            box.grid_remove()

    def _build(self):
        outer = ttk.Frame(self._make_scrollable_container(), padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(10, weight=1)

        title = ttk.Label(
            outer,
            text="다국어 자막 수집 · 언어별 하드자막 OCR",
            font=("Malgun Gothic", 16, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        source_modes = ttk.Frame(outer)
        source_modes.grid(row=1, column=0, columnspan=3, sticky="w")
        # Two rows keep four source modes readable on the default 820px window.
        source_choices = (
            *((("epsite", SITE_ADAPTER.SITE_LABEL),) if SITE_ADAPTER is not None else ()),
            ("local", "내 영상 1개 · 원본 자막 우선"),
            ("local-batch", "내 영상 최대 50개 일괄 처리"),
            ("captions", "웹 자막 우선 · YouTube·WeTV 등"),
        )
        for index, (value, title) in enumerate(source_choices):
            ttk.Radiobutton(
                source_modes, text=title, value=value,
                variable=self.source_mode_var,
                command=self._source_mode_changed,
            ).grid(row=index // 2, column=index % 2,
                   sticky="w", padx=(0, 20), pady=(0, 2))
        self.source_label = ttk.Label(
            outer,
            text=SITE_ADAPTER.URL_LABEL if SITE_ADAPTER is not None else "자막이 포함된 영상 파일",
        )
        self.source_label.grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(outer, textvariable=self.url_var).grid(
            row=2, column=1, sticky="ew", padx=(12, 8), pady=5
        )
        self.source_picker = ttk.Button(outer, text="영상 선택", command=self._pick_source, state="disabled")
        self.source_picker.grid(row=2, column=2, sticky="ew", pady=5)

        self.target_label = ttk.Label(
            outer,
            text="별도 고화질 영상 (선택 · 자막 재싱크용)",
        )
        self.target_label.grid(row=3, column=0, sticky="w", pady=5)
        self.target_entry = ttk.Entry(outer, textvariable=self.target_var)
        self.target_entry.grid(
            row=3, column=1, sticky="ew", padx=(12, 8), pady=5
        )
        self.target_picker = ttk.Button(outer, text="찾기", command=self._pick_target)
        self.target_picker.grid(
            row=3, column=2, sticky="ew", pady=5
        )

        # Explicit local subtitle selection is separate from EpSite target
        # video and public batch URL list. Only one group occupies grid row 3.
        self.sidecar_label = ttk.Label(outer, text="원본 자막 파일 (선택 · SRT/VTT/ASS)")
        self.sidecar_label.grid(row=3, column=0, sticky="w", pady=5)
        self.sidecar_entry = ttk.Entry(outer, textvariable=self.sidecar_var)
        self.sidecar_entry.grid(row=3, column=1, sticky="ew", padx=(12, 8), pady=5)
        self.sidecar_picker = ttk.Button(
            outer, text="자막 선택", command=self._pick_sidecar,
        )
        self.sidecar_picker.grid(row=3, column=2, sticky="ew", pady=5)
        for control in (self.sidecar_label, self.sidecar_entry, self.sidecar_picker):
            control.grid_remove()

        self.out_label = ttk.Label(outer, text="결과 파일 기준 경로")
        self.out_label.grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(outer, textvariable=self.out_var).grid(
            row=4, column=1, sticky="ew", padx=(12, 8), pady=5
        )
        ttk.Button(outer, text="저장 위치", command=self._pick_output).grid(
            row=4, column=2, sticky="ew", pady=5
        )

        self.languages_label = ttk.Label(
            outer, text="수집 언어 (쉼표 구분, 예: ko,en,ja,zh-Hans)"
        )
        self.languages_label.grid(row=5, column=0, sticky="w", pady=5)
        self.languages_entry = ttk.Entry(
            outer, textvariable=self.languages_var, state="disabled"
        )
        self.languages_entry.grid(row=5, column=1, sticky="ew", padx=(12, 8), pady=5)
        ocr_box = ttk.Frame(outer)
        ocr_box.grid(row=5, column=2, sticky="w", pady=5)
        ttk.Label(ocr_box, text="영상 OCR 언어").pack(side="left")
        self.ocr_language_entry = ttk.Combobox(
            ocr_box, textvariable=self.ocr_language_var,
            values=("ko", "en", "ja", "zh"),
            state="readonly", width=7,
        )
        self.ocr_language_entry.pack(side="left", padx=(6, 0))
        self.fast_cpu_check = ttk.Checkbutton(
            ocr_box,
            text="빠른 CPU (OpenVINO)",
            variable=self.fast_cpu_var,
            state="normal" if openvino_runtime_available() else "disabled",
        )
        self.fast_cpu_check.pack(side="left", padx=(10, 0))

        proxy_frame = ttk.LabelFrame(outer, text="웹 자막 프록시 (선택 · DomainVPN과 독립)", padding=8)
        proxy_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        proxy_frame.columnconfigure(1, weight=1)
        ttk.Label(proxy_frame, text="연결 방식").grid(row=0, column=0, sticky="w")
        modes = ttk.Frame(proxy_frame)
        modes.grid(row=0, column=1, columnspan=2, sticky="w")
        for code, title in (
            ("direct", "직접 연결"),
            ("required", "프록시 필수"),
            ("fallback", "직접 실패 시 프록시 재시도"),
        ):
            ttk.Radiobutton(
                modes, text=title, value=code,
                variable=self.proxy_mode_var, command=self._proxy_mode_changed,
            ).pack(side="left", padx=(0, 12))
        ttk.Label(proxy_frame, text="프록시 주소").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.proxy_entry = ttk.Entry(
            proxy_frame, textvariable=self.proxy_url_var, state="disabled",
        )
        self.proxy_entry.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(6, 0))
        ttk.Label(proxy_frame, text="예: socks5h://127.0.0.1:17920").grid(
            row=1, column=2, sticky="w", padx=(10, 0), pady=(6, 0),
        )
        self.video_ocr_check = ttk.Checkbutton(
            proxy_frame,
            text="선택한 언어의 공개 자막이 없으면 영상 OCR (최대 512MiB, 임시 영상 자동 삭제)",
            variable=self.video_ocr_var,
            state="disabled",
        )
        self.video_ocr_check.grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(8, 0),
        )

        ttk.Checkbutton(
            outer,
            text="상세 진단 저장 (선택 · OCR 화면/문구를 로컬에 저장)",
            variable=self.diag_var,
        ).grid(row=7, column=1, columnspan=2, sticky="w", padx=(12, 0), pady=(6, 4))

        # Optional / experimental tools stay available but collapsed so the
        # main flow (input -> start -> log) fits on one screen.
        self.show_advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            outer,
            text="고급·실험 도구 표시",
            variable=self.show_advanced_var,
            command=self._toggle_advanced_tools,
        ).grid(row=7, column=0, sticky="w", pady=(6, 4))

        runtime_box = ttk.LabelFrame(outer, text="런타임 상태 · 고급/실험 도구", padding=10)
        runtime_box.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        self._advanced_box = runtime_box
        runtime_box.columnconfigure(0, weight=1)
        self.runtime_label = ttk.Label(runtime_box, text="")
        self.runtime_label.grid(row=0, column=0, sticky="w")
        ttk.Button(
            runtime_box,
            text="자체검사",
            command=self._run_selftest_ui,
        ).grid(row=0, column=1, padx=(10, 0), sticky="e")
        self.native_vsf_check = ttk.Checkbutton(
            runtime_box,
            text="실험: 내 영상의 무료 VideoSubFinder 감지기 사용 (최초 공식 다운로드 별도 승인)",
            variable=self.native_vsf_var,
            command=self._refresh_native_split_state,
            state="disabled",
        )
        self.native_vsf_check.grid(
            row=1, column=0, sticky="w", pady=(8, 0),
        )
        self.native_vsf_release_choice = ttk.Combobox(
            runtime_box, textvariable=self.native_vsf_release_var,
            values=("v6.10.5", "v6.10.7"), state="readonly", width=10,
        )
        self.native_vsf_release_choice.grid(row=1, column=1, sticky="e", pady=(8, 0))
        self.native_split_check = ttk.Checkbutton(
            runtime_box,
            text="실험: 연속 대사 2개 분리 (증거가 충분할 때만 · 기본 끔)",
            variable=self.native_split_var, state="disabled",
        )
        self.native_split_check.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0),
        )

        review_controls = ttk.Frame(runtime_box)
        review_controls.grid(row=3, column=0, columnspan=2,
                             sticky="w", pady=(8, 0))
        self.review_button = ttk.Button(
            review_controls, text="원본 화면 검토 (수동)",
            command=self._start_frame_review, state="disabled",
        )
        self.review_button.pack(side="left")
        ttk.Label(review_controls, text="범위").pack(
            side="left", padx=(8, 3),
        )
        ttk.Combobox(
            review_controls, textvariable=self.review_mode_var,
            values=("경고 자막만", "추출 후 검토 권장",
                    "모든 자막 순서대로", "반복 워터마크 후보",
                    "연속 대사 분리 검토"),
            state="readonly", width=20,
        ).pack(side="left")
        ttk.Label(review_controls, text="이후 자막 번호").pack(
            side="left", padx=(8, 3),
        )
        ttk.Entry(
            review_controls, textvariable=self.review_after_var, width=6,
        ).pack(side="left")

        self.tab_capture_button = ttk.Button(
            runtime_box,
            text="실험: 브라우저 탭 화면에서 무료 자막 추출",
            command=self._start_tab_capture,
        )
        self.tab_capture_button.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.tab_capture_stop_button = ttk.Button(
            runtime_box, text="탭 녹화 안전 종료",
            command=self._stop_tab_capture, state="disabled",
        )
        self.tab_capture_stop_button.grid(row=4, column=1, sticky="e", pady=(8, 0))
        ttk.Checkbutton(
            runtime_box,
            text="무료 yt-dlp 자동 갱신 (기존 버전 보존 · 최초 승인)",
            variable=self.managed_caption_updates_var,
            command=self._toggle_managed_caption_updates,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Button(
            runtime_box, text="지금 확인",
            command=self._request_managed_caption_update,
        ).grid(row=5, column=1, sticky="e", pady=(8, 0))
        self.seconv54_check = ttk.Checkbutton(
            runtime_box,
            text="선택: 이미지 자막 변환기 SeConv 5.4 사용 (기본 5.3 유지 · 최초 승인)",
            variable=self.seconv54_var,
        )
        self.seconv54_check.grid(row=6,column=0,columnspan=2,sticky="w",pady=(8,0))

        preview_controls = ttk.Frame(runtime_box)
        preview_controls.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.ocr_preview_button = ttk.Button(
            preview_controls, text="내 영상 1프레임 OCR 미리보기",
            command=self._start_local_ocr_preview, state="disabled",
        )
        self.ocr_preview_button.pack(side="left")
        ttk.Label(preview_controls, text="시각 (시:분:초)").pack(
            side="left", padx=(9, 3),
        )
        ttk.Entry(preview_controls, textvariable=self.preview_time_var,
                  width=11).pack(side="left")
        ttk.Label(preview_controls, text="OCR 범위").pack(
            side="left", padx=(7, 3),
        )
        self.local_ocr_region_choice = ttk.Combobox(
            preview_controls, textvariable=self.local_ocr_region_var,
            values=tuple(_LOCAL_OCR_REGION_LABELS), state="disabled", width=28,
        )
        self.local_ocr_region_choice.pack(side="left")
        ttk.Label(preview_controls, text="원본 불변").pack(
            side="left", padx=(5, 0),
        )
        proofread_controls = ttk.Frame(runtime_box)
        proofread_controls.grid(row=8, column=0, columnspan=2,
                                sticky="w", pady=(7, 0))
        self.subtitle_edit_button = ttk.Button(
            proofread_controls,
            text="무료 교정 (Subtitle Edit / Windows 메모장)",
            command=self._start_external_proofread,
            state="disabled",
        )
        self.subtitle_edit_button.pack(side="left")
        self.proofread_ass_button = ttk.Button(
            proofread_controls,
            text="교정 SRT → 새 ASS 내보내기",
            command=self._export_external_proofread_ass,
            state="disabled",
        )
        self.proofread_ass_button.pack(side="left", padx=(8, 0))
        self.corrected_quality_button = ttk.Button(
            proofread_controls, text="교정 품질 비교",
            command=self._compare_corrected_quality, state="disabled",
        )
        self.corrected_quality_button.pack(side="left", padx=(8, 0))
        self.ocr_compare_button = ttk.Button(
            proofread_controls, text="기본↔밝은 OCR 비교",
            command=lambda: self._start_local_ocr_preview(compare_bright=True),
            state="disabled",
        )
        self.ocr_compare_button.pack(side="left", padx=(8, 0))
        self.focus_bottom_two_check = ttk.Checkbutton(
            runtime_box,
            text="실험: 하드자막 OCR이 3줄 이상 읽으면 아래 2줄만 사용"
                 " (진짜 3줄 자막 유실 위험 · 기본 끔)",
            variable=self.focus_bottom_two_var, state="disabled",
        )
        self.focus_bottom_two_check.grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )
        # Keep both optical alternatives in one row: the compact GUI
        # must not push the extraction button below the 820x760 viewport.
        timing_choices = ttk.Frame(runtime_box)
        timing_choices.grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )
        self.light_caption_temporal_check = ttk.Checkbutton(
            timing_choices,
            text="실험: 밝은 글자 + 시간 보정 (자막 유실 위험)",
            variable=self.light_caption_temporal_var, state="disabled",
        )
        self.light_caption_temporal_check.grid(
            row=0, column=0, sticky="w", padx=(0, 12),
        )
        self.pixel_timing_only_check = ttk.Checkbutton(
            timing_choices,
            text="실험: 원문 보존 + 시간만 보정 (CPU 재검사)",
            variable=self.pixel_timing_only_var, state="disabled",
        )
        self.pixel_timing_only_check.grid(
            row=0, column=1, sticky="w",
        )
        self.caption_islands_check = ttk.Checkbutton(
            runtime_box,
            text="실험: OCR의 위치 정보로 제목·워터마크와 대사 묶음 구분"
                 " (모호하면 보존 · 진짜 대사 누락 가능 · 기본 끔)",
            variable=self.caption_islands_var, state="disabled",
        )
        self.caption_islands_check.grid(
            row=11, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )

        buttons = ttk.Frame(outer)
        buttons.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        buttons.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(
            buttons,
            text="자막 추출 시작",
            command=self._start,
        )
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.cancel_button = ttk.Button(
            buttons, text="안전하게 취소", command=self._cancel,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            buttons,
            text="결과 폴더 열기",
            command=self._open_output_folder,
        ).grid(row=0, column=2, padx=(10, 0))
        self.progress_bar = ttk.Progressbar(
            buttons, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress_bar.grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )

        log_box = ttk.LabelFrame(outer, text="진행 로그", padding=8)
        log_box.grid(row=10, column=0, columnspan=3, sticky="nsew")
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)

        self.log = tk.Text(
            log_box, wrap="word", state="disabled", font=("Consolas", 9), height=12,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        ttk.Label(outer, textvariable=self.status_var).grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )
        self._toggle_advanced_tools()
        self._fit_window_to_screen(outer)

    def _log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _warn_runtime_missing(self, message: str):
        # The runtime box is collapsed by default; surface problems in the log.
        if getattr(self, "log", None) is None:
            return
        try:
            self._log("[경고] " + message)
        except Exception:
            pass

    def _refresh_runtime_status(self):
        parts = []
        try:
            parts.append(f"FFmpeg: {resolve_binary('ffmpeg')}")
            parts.append(f"FFprobe: {resolve_binary('ffprobe')}")
        except Exception as exc:
            parts.append(f"FFmpeg/FFprobe: 누락 ({exc})")
            self._warn_runtime_missing(f"FFmpeg/FFprobe를 찾지 못했습니다: {exc}")

        model_dir = bundled_model_dir()
        manifest = bundled_manifest_path()
        if model_dir.is_dir() and manifest.is_file():
            parts.append(f"OCR 모델: {model_dir}")
        else:
            parts.append("OCR 모델: 번들에 없음 (소프트 자막 회차는 사용 가능)")
            self._warn_runtime_missing(
                "OCR 모델이 없습니다. 하드자막 OCR을 쓰려면 설치 스크립트를 다시 실행하세요."
            )
        self.runtime_label.configure(text="\n".join(parts))

    def _run_selftest_ui(self):
        try:
            report = run_selftest(require_models=True)
            lines = [
                ("OK" if x.ok else "FAIL") + f" | {x.name} | {x.detail}"
                for x in report.checks
            ]
            self._log("자체검사")
            for line in lines:
                self._log(line)
            if report.ok:
                messagebox.showinfo("자체검사", "필수 런타임 검사를 모두 통과했습니다.")
            else:
                messagebox.showwarning(
                    "자체검사",
                    "일부 검사가 실패했습니다. 진행 로그를 확인하세요.",
                )
        except Exception as exc:
            messagebox.showerror("자체검사 실패", str(exc))

    def _toggle_managed_caption_updates(self):
        enabled = bool(self.managed_caption_updates_var.get())
        if enabled and not messagebox.askyesno(
            "공식 무료 도구 자동 갱신",
            "GitHub의 yt-dlp 공식 안정판만 확인합니다. 새 버전은 사용자 전용 폴더에서 "
            "SHA-256과 오프라인 실행 검증을 마친 뒤 선택합니다. 기존 HSE EXE와 "
            "기존 버전은 보존하며 추가 과금이나 유료 API는 없습니다.\n\n"
            "이후 HSE를 실행할 때 하루 한 번 자동 확인할까요?",
        ):
            self.managed_caption_updates_var.set(False)
            return
        try:
            set_auto_update_enabled(enabled)
        except Exception as exc:
            self.managed_caption_updates_var.set(not enabled)
            messagebox.showerror("업데이트 설정 오류", type(exc).__name__)
            return
        self._log("공식 yt-dlp 자동 갱신 " + ("활성화" if enabled else "비활성화 · 번들 버전 사용"))
        if enabled:
            self._start_managed_caption_update(auto=False)

    def _request_managed_caption_update(self):
        if not self.managed_caption_updates_var.get():
            self.managed_caption_updates_var.set(True)
            self._toggle_managed_caption_updates()
            return
        self._start_managed_caption_update(auto=False)

    def _start_managed_caption_update(self, *, auto: bool):
        if self._managed_caption_update_running:
            return
        self._managed_caption_update_running = True
        def worker():
            try:
                result = refresh_managed_downloader(auto=auto)
            except Exception as exc:
                # Never expose an upstream URL, filesystem path or credentials.
                result = "error:" + type(exc).__name__
            finally:
                self.events.put(("managed_caption_update_done", None))
            self.events.put(("managed_caption_update_status", result))
        threading.Thread(
            target=worker, name="HSE-OfficialDownloaderUpdate", daemon=True,
        ).start()

    def _start_tab_capture(self):
        # No tab is accessed until Chrome/Edge asks the user for explicit
        # screen sharing permission. Never reuse their browser profile.
        if getattr(self, "_pending_close", False):
            return
        current = getattr(self, "_tab_capture_thread", None)
        if ((current is not None and current.is_alive())
                or (self.worker is not None and self.worker.is_alive())):
            return
        try:
            language=normalize_ocr_language(self.ocr_language_var.get())
            raw=self.out_var.get().strip()
            if not raw:
                raise ValueError("자막 저장 위치를 선택하세요.")
            selected=Path(raw).expanduser()
            folder=(selected if self.source_mode_var.get() in
                    {"captions","local-batch"} else selected.parent)
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("기존 자막 저장 폴더를 선택하세요.")
            from .model_assets import resolve_model_paths
            resolve_model_paths(
                bundled_model_dir(), manifest_path=bundled_manifest_path(),
                language=language, verify=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("브라우저 화면 추출 준비", _user_error_summary(exc))
            return
        if not messagebox.askyesno(
            "브라우저 화면 공유 동의",
            "Chrome/Edge에서 영상이 보이는 탭을 직접 선택해야 녹화됩니다. "
            "영상·OCR은 PC 안에서만 처리되며 추가 유료 서비스나 "
            "기존 파일 덮어쓰기는 없습니다.\n\n"
            "이어 열리는 브라우저 페이지에서 원본 시작 시각·"
            "자동 종료 시간·OCR 방식·자막 영역을 지정할 수 있습니다. "
            "기본은 하단 40%이며 더 넓게 선택하면 화면 글자 오인식과 "
            "처리 시간 증가 위험이 있습니다. 자동 종료는 녹화 시간 "
            "기준이므로 일시정지·광고에 자동 대응하지 않습니다. "
            "보호 영상의 검은 화면은 복구하지 않습니다. "
            "연구 기능을 시작하시겠습니까?",
        ):
            return
        self._tab_capture_stop=threading.Event()
        self._tab_capture_outcome_received=False
        self.tab_capture_button.configure(state="disabled")
        self.tab_capture_stop_button.configure(state="normal")
        self.start_button.configure(state="disabled")
        if getattr(self, "review_button", None) is not None:
            self.review_button.configure(state="disabled")
        self.status_var.set("브라우저 탭 선택 대기")
        def launch():
            try:
                from .tab_capture import serve
                serve(
                    folder, open_browser=True, ocr_language=language,
                    stop_event=self._tab_capture_stop,
                    on_ready=lambda url: self.events.put(("tab_capture_ready", url)),
                    on_finished=lambda result: self.events.put(
                        ("tab_capture_result", result)
                    ),
                )
            except Exception as exc:
                self.events.put(("tab_capture_error", type(exc).__name__))
            finally:
                self.events.put(("tab_capture_closed", None))
        self._tab_capture_thread=threading.Thread(target=launch,daemon=True)
        self._tab_capture_thread.start()

    def _stop_tab_capture(self):
        job=getattr(self, "_tab_capture_thread", None)
        if job is None or not job.is_alive():
            return
        self._tab_capture_stop.set()
        self.tab_capture_stop_button.configure(state="disabled")
        self.status_var.set("탭 녹화·OCR 안전 종료 중")

    def _proxy_mode_changed(self):
        self.proxy_entry.configure(
            state="normal" if self.source_mode_var.get() == "captions"
            and self.proxy_mode_var.get() != "direct" else "disabled"
        )

    def _refresh_native_split_state(self):
        enabled = (
            self.source_mode_var.get() in {"local", "local-batch"}
            and bool(self.native_vsf_var.get())
        )
        if not enabled:
            self.native_split_var.set(False)
        if getattr(self, "native_split_check", None) is not None:
            self.native_split_check.configure(
                state="normal" if enabled else "disabled",
            )

    def _source_mode_changed(self):
        mode = self.source_mode_var.get()
        self._local_batch_selection = ()
        self._local_batch_display = ""
        self._requested_local_batch_paths = ()
        local = mode == "local"
        local_batch = mode == "local-batch"
        native_allowed = local or local_batch
        if getattr(self, "local_ocr_region_var", None) is not None and not native_allowed:
            self.local_ocr_region_var.set("기본 · 하단 40%")
        if getattr(self, "local_ocr_region_choice", None) is not None:
            self.local_ocr_region_choice.configure(
                state="readonly" if native_allowed else "disabled",
            )
        if getattr(self, "native_vsf_var", None) is not None and not native_allowed:
            self.native_vsf_var.set(False)
        if getattr(self, "native_vsf_check", None) is not None:
            self.native_vsf_check.configure(
                state="normal" if native_allowed else "disabled",
            )
        if getattr(self, "native_split_var", None) is not None:
            self._refresh_native_split_state()
        if getattr(self, "focus_bottom_two_var", None) is not None and not native_allowed:
            self.focus_bottom_two_var.set(False)
        if getattr(self, "focus_bottom_two_check", None) is not None:
            self.focus_bottom_two_check.configure(
                state="normal" if native_allowed else "disabled"
            )
        if getattr(self, "caption_islands_var", None) is not None and not native_allowed:
            self.caption_islands_var.set(False)
        if getattr(self, "caption_islands_check", None) is not None:
            self.caption_islands_check.configure(
                state="normal" if native_allowed else "disabled"
            )
        if getattr(self, "light_caption_temporal_var", None) is not None and not native_allowed:
            self.light_caption_temporal_var.set(False)
        if getattr(self, "light_caption_temporal_check", None) is not None:
            self.light_caption_temporal_check.configure(
                state="normal" if native_allowed else "disabled"
            )
        if getattr(self, "pixel_timing_only_var", None) is not None and not native_allowed:
            self.pixel_timing_only_var.set(False)
        if getattr(self, "pixel_timing_only_check", None) is not None:
            self.pixel_timing_only_check.configure(
                state="normal" if native_allowed else "disabled"
            )
        if getattr(self, "review_button", None) is not None:
            worker = getattr(self, "worker", None)
            busy = worker is not None and worker.is_alive()
            self.review_button.configure(
                state="normal" if local and not busy else "disabled",
            )
        if getattr(self, "ocr_preview_button", None) is not None:
            self.ocr_preview_button.configure(
                state="normal" if local and not (
                    getattr(self, "worker", None)
                    and self.worker.is_alive()
                ) else "disabled"
            )
        if getattr(self, "ocr_compare_button", None) is not None:
            self.ocr_compare_button.configure(
                state="normal" if local and not (
                    getattr(self, "worker", None)
                    and self.worker.is_alive()
                ) else "disabled"
            )
        if getattr(self, "subtitle_edit_button", None) is not None:
            self.subtitle_edit_button.configure(
                state="normal" if local and not (
                    getattr(self, "worker", None)
                    and self.worker.is_alive()
                ) else "disabled"
            )
        if getattr(self, "proofread_ass_button", None) is not None:
            self.proofread_ass_button.configure(
                state="normal" if local and not (
                    getattr(self, "worker", None)
                    and self.worker.is_alive()
                ) else "disabled"
            )
        if getattr(self, "corrected_quality_button", None) is not None:
            self.corrected_quality_button.configure(
                state="normal" if local and not (
                    getattr(self, "worker", None)
                    and self.worker.is_alive()
                ) else "disabled"
            )
        captions = mode == "captions"
        self.url_var.set("")
        reset_target = getattr(self.target_var, "set", None)
        if callable(reset_target):
            reset_target("")
        if getattr(self, "sidecar_var", None) is not None:
            self.sidecar_var.set("")
        self.source_label.configure(
            text="동영상 여러 개 선택 또는 경로 목록(.txt 직접 입력)" if local_batch
            else "자막이 포함된 영상 파일" if local
            else "공개 영상 URL (YouTube·WeTV·Vimeo·Dailymotion)" if captions
            else getattr(SITE_ADAPTER, "URL_LABEL", "")
        )
        self.source_picker.configure(
            state="normal" if local or local_batch else "disabled",
            text="영상 여러 개 선택" if local_batch else "영상 선택",
        )
        for control in (self.target_entry, self.target_picker):
            control.configure(state="disabled" if local or local_batch else "normal")
        if all(getattr(self, attr, None) is not None for attr in (
            "target_label", "sidecar_label", "sidecar_entry", "sidecar_picker",
        )):
            if local:
                for control in (self.target_label, self.target_entry, self.target_picker):
                    control.grid_remove()
                for control in (self.sidecar_label, self.sidecar_entry, self.sidecar_picker):
                    control.grid()
            elif local_batch:
                for control in (
                    self.sidecar_label, self.sidecar_entry, self.sidecar_picker,
                    self.target_label, self.target_entry, self.target_picker,
                ):
                    control.grid_remove()
            else:
                for control in (self.sidecar_label, self.sidecar_entry, self.sidecar_picker):
                    control.grid_remove()
                for control in (self.target_label, self.target_entry, self.target_picker):
                    control.grid()
        if getattr(self, "target_label", None) is not None:
            self.target_label.configure(
                text="회차 URL 목록(.txt, 선택 · 한 줄에 하나)" if captions
                else "별도 고화질 영상 (선택 · 자막 재싱크용)"
            )
        if getattr(self, "out_label", None) is not None:
            self.out_label.configure(
                text="자막 저장 폴더 (기존 결과 보존)" if captions or local_batch
                else "결과 파일 기준 경로"
            )
        if getattr(self, "languages_entry", None) is not None:
            self.languages_entry.configure(state="normal" if captions else "disabled")
        if getattr(self, "proxy_entry", None) is not None:
            self._proxy_mode_changed()
        if getattr(self, "video_ocr_check", None) is not None:
            self.video_ocr_check.configure(state="normal" if captions else "disabled")
        if getattr(self, "ocr_language_entry", None) is not None:
            self.ocr_language_entry.configure(
                state="disabled" if captions else "readonly"
            )
        if captions:
            self.target_picker.configure(text="목록 선택")
        else:
            self.target_picker.configure(text="찾기")

    def _pick_source(self):
        video_types = [
            ("Video", "*.mkv *.mp4 *.webm *.ogv *.avi *.mov *.m4v *.ts"),
            ("All files", "*.*"),
        ]
        if self.source_mode_var.get() == "local-batch":
            # Native Explorer multi-select replaces manual path-list creation
            # for the normal 1..50-episode workflow. A typed .txt list
            # remains supported for users who already maintain one.
            paths = filedialog.askopenfilenames(
                title="처리할 동영상 1~50개 선택",
                filetypes=video_types,
            )
            if not paths:
                return
            try:
                items = load_local_batch(
                    paths, None, Path(self.out_var.get()).expanduser(),
                )
            except (ValueError, OSError):
                messagebox.showerror(
                    "동영상 확인", "선택한 영상 목록이 유효하지 않습니다."
                    " 지원하는 영상 파일 1~50개만 선택하세요.",
                )
                return
            selected = tuple(str(item.source) for item in items)
            display = f"선택한 동영상 {len(selected)}개 (목록 파일 불필요)"
            self._local_batch_selection = selected
            self._local_batch_display = display
            self.url_var.set(display)
            return
        path = filedialog.askopenfilename(
            title="자막이 포함된 영상 파일 선택", filetypes=video_types,
        )
        if path:
            self.url_var.set(path)

    def _pick_sidecar(self):
        path = filedialog.askopenfilename(
            title="원본 텍스트 자막 선택 (직접 선택한 파일이 OCR보다 우선)",
            filetypes=[
                ("Subtitles", "*.srt *.vtt *.ass *.ssa"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.sidecar_var.set(path)

    def _pick_target(self):
        if self.source_mode_var.get() == "captions":
            path = filedialog.askopenfilename(
                title="한 줄에 회차 URL 하나씩 적힌 UTF-8 텍스트 파일",
                filetypes=[("URL list", "*.txt"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="자막을 재싱크할 별도 고화질 영상 선택",
                filetypes=[
                    ("Video", "*.mkv *.mp4 *.webm *.ogv *.avi *.mov *.m4v"),
                    ("All files", "*.*"),
                ],
            )
        if path:
            self.target_var.set(path)

    def _pick_output(self):
        initial = Path(self.out_var.get()).expanduser()
        folder = filedialog.askdirectory(
            title="결과 저장 폴더",
            initialdir=str(initial.parent if initial.parent.exists() else Path.home()),
        )
        if folder:
            if self.source_mode_var.get() in {"captions", "local-batch"}:
                self.out_var.set(str(Path(folder)))
            else:
                self.out_var.set(str(Path(folder) / "extracted-subtitles"))

    def _open_output_folder(self):
        out = Path(self.out_var.get()).expanduser()
        path = out if self.source_mode_var.get() in {"captions", "local-batch"} else out.parent
        try:
            import os
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("결과 폴더", str(path))
        except Exception as exc:
            messagebox.showerror("오류", str(exc))

    def _validate(self) -> tuple[str, str, str | None, Path]:
        mode = self.source_mode_var.get()
        source = self.url_var.get().strip()
        if mode == "local":
            path = Path(source).expanduser()
            if not source or not path.is_file():
                raise ValueError("자막이 화면에 포함된 영상 파일을 선택하세요.")
            source = str(path)
            raw_selected = (
                self.sidecar_var.get().strip()
                if getattr(self, "sidecar_var", None) is not None else ""
            )
            if raw_selected:
                chosen = Path(raw_selected).expanduser()
                if (chosen.is_symlink() or not chosen.is_file()
                        or chosen.suffix.lower() not in {".srt", ".vtt", ".ass", ".ssa"}):
                    raise ValueError("선택한 원본 자막은 기존 SRT/VTT/ASS/SSA 파일이어야 합니다.")
                target = str(chosen)
            else:
                target = None
        elif mode == "local-batch":
            selected = tuple(getattr(self, "_local_batch_selection", ()))
            display = getattr(self, "_local_batch_display", "")
            destination = Path(self.out_var.get()).expanduser()
            if selected and source == display:
                # Never parse a displayed count into paths. Only the private
                # canonical selections captured by Explorer are executable.
                load_local_batch(selected, None, destination)
                self._requested_local_batch_paths = selected
            else:
                chosen = Path(source).expanduser()
                if chosen.is_symlink() or chosen.suffix.lower() != ".txt":
                    raise ValueError(
                        "동영상 여러 개를 선택하거나 UTF-8 .txt 경로 목록을 입력하세요."
                    )
                # Validate before launching the worker, not after writing.
                load_local_batch([], chosen, destination)
                source = str(chosen)
                self._requested_local_batch_paths = ()
            target = None
        elif mode == "epsite":
            from .cli import _is_epsite_url
            if SITE_ADAPTER is None:
                raise ValueError("이 빌드에는 사이트 회차 어댑터가 포함되어 있지 않습니다.")
            if not _is_epsite_url(source):
                raise ValueError("올바른 사이트 회차 URL을 입력하세요.")
            target = self.target_var.get().strip() or None
            if target and not Path(target).is_file():
                raise ValueError("선택한 고화질 영상 파일을 찾을 수 없습니다.")
        elif mode == "captions":
            target = self.target_var.get().strip() or None
            load_episode_urls([source], target)
            lang_text = (
                self.languages_var.get()
                if getattr(self, "languages_var", None) is not None else "ko"
            )
            langs = [normalize_language(code) for code in lang_text.split(",")]
            if not langs or len(langs) > 6 or len(langs) != len(set(langs)):
                raise ValueError("언어는 최대 6개까지 중복 없이 선택하세요.")
            self._requested_languages = tuple(langs)
            proxy_mode = self.proxy_mode_var.get() if getattr(self, "proxy_mode_var", None) is not None else "direct"
            proxy_url = self.proxy_url_var.get().strip() if getattr(self, "proxy_url_var", None) is not None else ""
            self._caption_proxy_route = CaptionProxyRoute(proxy_mode, proxy_url if proxy_mode != "direct" else "")
            self._requested_video_ocr_enabled = bool(
                self.video_ocr_var.get() if getattr(self, "video_ocr_var", None) is not None else False
            )
        else:
            raise ValueError("입력 방식을 선택하세요.")

        if mode in {"epsite", "local", "local-batch"}:
            requested = (self.ocr_language_var.get() if getattr(self, "ocr_language_var", None) is not None else "ko")
            self._requested_ocr_language = normalize_ocr_language(requested)
            fast_cpu = bool(
                self.fast_cpu_var.get()
                if getattr(self, "fast_cpu_var", None) is not None else False
            )
            self._requested_ocr_backend = (
                "auto" if fast_cpu else "onnxruntime"
            )

        out = Path(self.out_var.get()).expanduser()
        if not out.name:
            raise ValueError("결과 파일 경로를 지정하세요.")
        if mode in {"captions", "local-batch"}:
            if (out.exists() and not out.is_dir()) or out.is_symlink():
                raise ValueError("자막 저장 위치는 파일이 아닌 폴더여야 합니다.")
        else:
            _check_no_existing_subtitles(
                out, target_sync=(mode == "epsite" and bool(target))
            )
            if mode == "local":
                for extra in _all_local_outputs(out):
                    if extra.exists() or extra.is_symlink():
                        raise ValueError(
                            "이미 같은 이름의 내장자막 출처 결과가 있습니다. "
                            "기존 파일을 보존하려면 다른 결과 경로를 지정하세요."
                        )
            out.parent.mkdir(parents=True, exist_ok=True)
        return mode, source, target, out

    def _make_rapidocr_adapter(
        self, *, model_dir: str, manifest_path: str, language: str,
        inference_engine: str | None = None,
    ) -> RapidOCRAdapter:
        backend = (
            inference_engine
            if inference_engine is not None
            else getattr(self, "_active_ocr_backend", "onnxruntime")
        )
        return RapidOCRAdapter(
            model_dir=model_dir,
            manifest_path=manifest_path,
            language=language,
            inference_engine=backend,
        )

    def _make_monitor(self) -> ExtractionMonitor | None:
        # Older headless GUI unit tests construct the GUI without __init__.
        # Those calls intentionally retain the established production API.
        stop = getattr(self, "_cancel_event", None)
        if stop is None:
            return None
        return ExtractionMonitor(
            on_progress=lambda event: self.events.put(("progress", event)),
            is_cancelled=stop.is_set,
            exclusive_outputs=True,
        )

    def _cancel(self):
        if self.worker is None or not self.worker.is_alive():
            return
        self._cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("취소 요청됨 · 안전한 중단 지점까지 진행")
        self._log("취소 요청을 전달했습니다. 현재 작업 단계가 중단되는 중입니다.")

    def _display_progress(self, event: ProgressEvent):
        labels = {
            "resolving": "회차/자막 소스 확인",
            "capturing": "영상에서 자막 화면 수집",
            "captured": "프레임 수집 완료",
            "recognizing": "영상 자막 OCR 인식",
            "exporting": "자막 파일 저장",
            "syncing": "별도 영상판 자막 싱크",
            "completed": "처리 완료",
        }
        label = labels.get(event.stage, "처리 중")
        if event.stage != self._progress_stage:
            self._progress_stage = event.stage
            self._log("진행 단계: " + label)
        if event.stage == "capturing":
            if self.progress_bar.cget("mode") != "indeterminate":
                self.progress_bar.configure(mode="indeterminate", value=0)
                self.progress_bar.start(12)
            suffix = f" · 영상 {event.media_seconds:.1f}초" if event.media_seconds is not None else ""
            summary = f"{label} · {event.completed:,}프레임{suffix}"
        elif event.stage == "recognizing":
            if event.total is None:
                # Long-form streaming cannot know the final OCR count yet.
                # Never display a misleading x/0 progress denominator.
                if self.progress_bar.cget("mode") != "indeterminate":
                    self.progress_bar.configure(mode="indeterminate", value=0)
                    self.progress_bar.start(12)
                clock = (
                    f" · 영상 {event.media_seconds:.1f}초"
                    if event.media_seconds is not None else ""
                )
                summary = f"{label} · {event.completed:,}관측{clock}"
            else:
                self.progress_bar.stop()
                self.progress_bar.configure(
                    mode="determinate", maximum=max(1, event.total),
                    value=event.completed,
                )
                summary = f"{label} · {event.completed:,}/{event.total:,}관측"
        else:
            self.progress_bar.stop()
            if event.stage == "completed":
                self.progress_bar.configure(mode="determinate", maximum=100, value=100)
            elif event.stage == "captured":
                self.progress_bar.configure(mode="determinate", maximum=100, value=0)
            summary = label
        if not self._cancel_event.is_set() or event.stage == "completed":
            self.status_var.set(summary)

    def _on_close(self):
        active_tab=getattr(self, "_tab_capture_thread", None)
        if active_tab is not None and active_tab.is_alive():
            self._pending_close=True
            self._tab_capture_stop.set()
            self.tab_capture_stop_button.configure(state="disabled")
            self.start_button.configure(state="disabled")
            self.status_var.set("현재 녹화·OCR를 안전하게 마친 뒤 종료")
            return
        # Tk may be destroyed only after browser/FFmpeg iterators and OCR have
        # unwound. Killing a daemon thread here risks orphaned child processes
        # and incomplete exported subtitles.
        if getattr(self, "_pending_close", False):
            return
        if self.worker is not None and self.worker.is_alive():
            self._pending_close = True
            self._cancel_event.set()
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="disabled")
            self.status_var.set("창 종료 요청됨 · 현재 추출을 안전하게 중단 중")
            self._log("창 닫기를 요청했습니다. 작업이 정리되면 자동으로 닫힙니다.")
            return
        self.root.destroy()

    def _start(self):
        if getattr(self, "_pending_close", False):
            return
        active_tab=getattr(self, "_tab_capture_thread", None)
        if active_tab is not None and active_tab.is_alive():
            return
        if self.worker and self.worker.is_alive():
            return
        try:
            args = self._validate()
        except Exception as exc:
            messagebox.showerror("입력 확인", str(exc))
            return

        native_choice = (
            bool(self.native_vsf_var.get())
            if args[0] in {"local", "local-batch"} and getattr(self, "native_vsf_var", None) is not None
            else False
        )
        native_split_choice = (
            bool(self.native_split_var.get())
            if native_choice and getattr(self, "native_split_var", None) is not None
            else False
        )
        focus_choice = (
            bool(self.focus_bottom_two_var.get())
            if args[0] in {"local", "local-batch"}
            and getattr(self, "focus_bottom_two_var", None) is not None else False
        )
        if focus_choice and native_choice:
            messagebox.showerror(
                "하드자막 OCR 실험 옵션",
                "하단 2줄 집중 옵션은 기본 HSE 영상 OCR에서만 동작합니다. "
                "VideoSubFinder 실험 옵션과 동시에 선택할 수 없습니다.",
            )
            return
        light_choice = (
            bool(self.light_caption_temporal_var.get())
            if args[0] in {"local", "local-batch"}
            and getattr(self, "light_caption_temporal_var", None) is not None else False
        )
        if light_choice and (native_choice or focus_choice):
            messagebox.showerror(
                "자막 광학 보정 실험 옵션",
                "밝은 글씨·어두운 배경 전용 광학 보정은 기본 영상 OCR에서만"
                " 사용할 수 있습니다. VideoSubFinder 및 하단 2줄 집중 옵션과"
                " 동시에 사용할 수 없습니다.",
            )
            return
        islands_choice = (
            bool(self.caption_islands_var.get())
            if args[0] in {"local", "local-batch"}
            and getattr(self, "caption_islands_var", None) is not None else False
        )
        if islands_choice and (native_choice or focus_choice or light_choice):
            messagebox.showerror(
                "자막 글자 묶음 실험 옵션",
                "글자 묶음 필터는 기본 영상 OCR에서만 사용합니다. "
                "VideoSubFinder·하단 2줄·광학 시각 보정과 동시에 선택할 수 없습니다.",
            )
            return
        pixel_choice = (
            bool(self.pixel_timing_only_var.get())
            if args[0] in {"local", "local-batch"}
            and getattr(self, "pixel_timing_only_var", None) is not None else False
        )
        if pixel_choice and (native_choice or focus_choice or light_choice or islands_choice):
            messagebox.showerror(
                "픽셀 시간 보정 실험 옵션",
                "자막 개수를 보존하는 픽셀 시간 보정은 기본 HSE OCR과만 사용할 수 있습니다. "
                "다른 실험 OCR 옵션을 해제하세요.",
            )
            return
        # Validate and freeze the chosen ROI on the UI thread. Worker or
        # batch input changes after Start must never affect in-flight videos.
        try:
            value = (
                self.local_ocr_region_var.get()
                if args[0] in {"local", "local-batch"}
                and getattr(self, "local_ocr_region_var", None) is not None
                else "기본 · 하단 40%"
            )
            region_mode = _local_region_from_label(value)
            from .tab_capture import roi_for_region
            roi_for_region(region_mode)
        except (TypeError, ValueError):
            messagebox.showerror("OCR 인식 범위", "지원되는 화면 인식 범위를 선택하세요.")
            return
        if native_choice and region_mode != "bottom40":
            messagebox.showerror(
                "OCR 인식 범위",
                "VideoSubFinder는 자체 감지 범위를 사용합니다. "
                "기본 HSE OCR로 전환하거나 기본 하단 40%를 선택하세요.",
            )
            return
        selected_native_release = (
            self.native_vsf_release_var.get()
            if native_choice and getattr(self, "native_vsf_release_var", None) is not None
            else "v6.10.5"
        )
        if selected_native_release not in {"v6.10.5", "v6.10.7"}:
            messagebox.showerror("외부 감지기 버전 오류", "검증된 공식 버전만 사용할 수 있습니다.")
            return
        # Consent is requested while still on the Tk UI thread. A declined
        # confirmation starts NO worker and downloads no third-party code.
        if native_choice and not messagebox.askyesno(
            "무료 외부 감지기 실행 승인",
            f"VideoSubFinder CLI {selected_native_release} (GPL-2.0)를 사용합니다.\n\n"
            "선택한 영상에 원본 자막이 없을 때만 공식 GitHub 배포 ZIP을 "
            "검증하여 사용자 개인 캐시에 내려받고 실행합니다. "
            "영상은 외부로 업로드하지 않습니다.\n\n"
            "실험 기능이며 OCR 결과의 정확도는 검증이 필요합니다. 계속할까요?",
        ):
            return
        image_candidate = (
            bool(self.seconv54_var.get())
            if args[0] in {"local", "local-batch"} and getattr(self, "seconv54_var", None) is not None
            else False
        )
        if image_candidate and not messagebox.askyesno(
            "무료 이미지 자막 변환기 승인",
            "SeConv 5.4.0의 공식 SHA-256 고정 배포판을 사용합니다.\n\n"
            "원본 텍스트 자막이 없고 PGS/VobSub 이미지 자막이 확인될 때만 "
            "개인 캐시로 별도 다운로드하여 검증 후 실행합니다. 기존 5.3과 "
            "사용자 영상·자막은 변경하지 않습니다. 계속할까요?",
        ):
            return
        temporal_opt_in = (
            args[0] == "local"
            and getattr(self, "review_mode_var", None) is not None
            and self.review_mode_var.get() == "반복 워터마크 후보"
        )
        if temporal_opt_in and native_choice:
            messagebox.showerror(
                "검토 기능 선택 확인",
                "시간축 후보 검토는 기본 HSE 영상 OCR에서만 사용할 수 있습니다. "
                "외부 VideoSubFinder 연구 옵션을 끄고 다시 시작하세요.",
            )
            return
        self._active_temporal_review_opt_in = temporal_opt_in
        self._temporal_review_hints = None
        self._native_split_review_hints = None
        self._active_native_vsf_opt_in = native_choice
        self._active_native_split_opt_in = native_split_choice
        self._active_focus_bottom_two = focus_choice
        self._active_local_ocr_region = region_mode
        self._active_caption_islands = islands_choice
        self._active_light_caption_temporal = light_choice
        self._active_pixel_timing_only = pixel_choice
        self._active_seconv54_opt_in = image_candidate
        self._active_native_vsf_release = selected_native_release
        self._active_ocr_backend = getattr(
            self, "_requested_ocr_backend", "onnxruntime"
        )
        self._last_local_ocr_quality_signals = 0
        self._last_local_quality_samples = ()
        self._quality_review_hints = None
        self._last_batch_quality_review = (0, 0, 0)
        self._cancel_event.clear()
        self._progress_stage = None
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", maximum=100, value=0)
        self.start_button.configure(state="disabled")
        if getattr(self, "review_button", None) is not None:
            self.review_button.configure(state="disabled")
        if getattr(self, "ocr_preview_button", None) is not None:
            self.ocr_preview_button.configure(state="disabled")
        if getattr(self, "ocr_compare_button", None) is not None:
            self.ocr_compare_button.configure(state="disabled")
        if getattr(self, "subtitle_edit_button", None) is not None:
            self.subtitle_edit_button.configure(state="disabled")
        if getattr(self, "proofread_ass_button", None) is not None:
            self.proofread_ass_button.configure(state="disabled")
        if getattr(self, "corrected_quality_button", None) is not None:
            self.corrected_quality_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("작업 중")
        self._log("=" * 64)
        self._log("작업 시작")
        if getattr(self, "_active_ocr_backend", "onnxruntime") == "openvino":
            self._log(
                "빠른 CPU(OpenVINO) · 동일 SHA 고정 OCR 모델 사용 · "
                "원본 자막 우선 · 별도 유료 서비스 없음"
            )
        diagnostics_enabled = bool(self.diag_var.get())
        # The input widgets may change after the worker has started. Keep
        # terminal labels tied to the actual running source type.
        self._active_source_mode = args[0]
        languages = self._requested_languages if args[0] == "captions" else None
        ocr_language = self._requested_ocr_language if args[0] != "captions" else "ko"
        # Freeze the entire selection at Start. Switching GUI modes while a
        # worker is active must never redirect the running job to other media.
        local_batch_paths = (
            tuple(getattr(self, "_requested_local_batch_paths", ()))
            if args[0] == "local-batch" else ()
        )
        self.worker = threading.Thread(
            target=self._worker,
            args=(*args, diagnostics_enabled, languages, ocr_language, local_batch_paths),
            daemon=True,
        )
        self.worker.start()

    def _start_external_proofread(self):
        """Explicit choice of installed free Subtitle Edit or built-in Notepad.

        Reuse the same never-overwrite -proofread.srt name, numeric comparison
        and separate ASS export. No additional download or background action.
        """
        if getattr(self, "_pending_close", False):
            return
        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            return
        tab = getattr(self, "_tab_capture_thread", None)
        if tab is not None and tab.is_alive():
            return
        try:
            from .subtitle_edit_handoff import (
                find_existing_subtitle_edit, find_windows_notepad,
                validate_handoff, validate_proofread_source,
                launch_subtitle_edit_proofreading, launch_notepad_proofreading,
            )
            if self.source_mode_var.get() != "local":
                raise ValueError("requires one local video")
            video = Path(self.url_var.get().strip()).expanduser()
            srt = Path(self.out_var.get().strip()).expanduser().with_suffix(".srt")
            exe = find_existing_subtitle_edit()
            using_notepad = exe is None
            if using_notepad:
                exe = find_windows_notepad()
                if exe is None:
                    messagebox.showinfo(
                        "무료 자막 교정기",
                        "Subtitle Edit와 Windows 메모장을 찾지 못했습니다. "
                        "새 프로그램은 다운로드하지 않으며 HSE 원본은 변경하지 않습니다.",
                    )
                    return
                video, srt = validate_proofread_source(video, srt)
                editor = "Windows 메모장"
                limits = (
                    "메모장은 영상 동기화 기능이 없으므로 앞서 만든 "
                    "원본 화면 HTML을 참고해 SRT 텍스트·시각을 수정하세요."
                )
            else:
                video, srt, exe = validate_handoff(video, srt, executable=exe)
                editor = "Subtitle Edit"
                limits = "영상과 교정 SRT를 함께 엽니다."
            working = srt.with_name(srt.stem + "-proofread.srt")
            exists = working.exists() or working.is_symlink()
            question = (
                "기존 교정본을 그대로 엽니다. 저장된 사람의 수정은 덮어쓰지 않습니다."
                if exists else
                "최초 교정용 '-proofread.srt' 사본을 생성하고 원본을 보존합니다."
            )
            if not messagebox.askyesno(
                f"{editor} 교정 시작",
                question + "\n\n" + limits
                + "\n\n교정 SRT를 저장한 뒤 HSE에서 품질 비교와 "
                  "새 ASS 내보내기를 선택할 수 있습니다. 계속할까요?",
            ):
                return
            receipt = (
                launch_notepad_proofreading(video, srt, executable=exe)
                if using_notepad else
                launch_subtitle_edit_proofreading(video, srt, executable=exe)
            )
            self.status_var.set(f"별도 {editor} 교정 화면 실행")
            self._log(
                "무료 교정기 실행 · 원본 SRT/ASS·영상 보존 · "
                + ("교정 사본 생성" if receipt.newly_created
                   else "기존 교정 사본 재사용")
            )
        except Exception as exc:
            # Never expose source names, corrected text or native process args.
            self._log("무료 교정 화면 실행 실패 · " + type(exc).__name__)
            messagebox.showerror(
                "무료 자막 교정기",
                "원본 영상·SRT 또는 교정 사본·무료 편집기 상태를 확인하세요. "
                "원본 영상과 자막은 변경하지 않았습니다.",
            )

    def _export_external_proofread_ass(self):
        """An explicit, non-overwriting export after human proofreading."""
        if getattr(self, "_pending_close", False):
            return
        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            return
        tab = getattr(self, "_tab_capture_thread", None)
        if tab is not None and tab.is_alive():
            return
        try:
            if self.source_mode_var.get() != "local":
                raise ValueError("requires one local video")
            from .subtitle_edit_handoff import export_proofread_ass
            original = Path(self.out_var.get().strip()).expanduser().with_suffix(".srt")
            result = export_proofread_ass(original)
            self.status_var.set(
                "교정 ASS 기존 결과 확인" if result.already_present
                else "교정 ASS 새로 생성 완료"
            )
            self._log(
                "교정 SRT → ASS 내보내기 · 원본 SRT/ASS 보존 · "
                + ("이미 같은 내용 있음" if result.already_present
                   else "새 ASS 생성")
            )
            # File location is shown ONLY in a local dialog, not the log.
            messagebox.showinfo(
                "교정 자막 ASS",
                ("기존 ASS를 그대로 재사용합니다.\n"
                 if result.already_present else
                 "교정 내용의 새 ASS를 만들었습니다.\n")
                + str(result.ass_path)
                + "\n\n원본 자막과 교정 SRT는 변경하지 않았습니다.",
            )
        except Exception as exc:
            self._log("교정 ASS 내보내기 실패 · " + type(exc).__name__)
            messagebox.showerror(
                "교정 자막 ASS",
                "교정 SRT가 없거나 완전하지 않거나 ASS 출력이 충돌합니다. "
                "원본과 교정 사본은 보존됐습니다.",
            )

    def _compare_corrected_quality(self):
        """Measure edited local SRT against unmodified HSE SRT, no writes."""
        if getattr(self, "_pending_close", False):
            return
        worker = getattr(self, "worker", None)
        tab = getattr(self, "_tab_capture_thread", None)
        if (worker is not None and worker.is_alive()
                or tab is not None and tab.is_alive()):
            return
        try:
            if self.source_mode_var.get() != "local":
                raise ValueError("one local video mode required")
            from .corrected_quality import audit_proofread_srt
            original = Path(self.out_var.get().strip()).expanduser().with_suffix(".srt")
            score = audit_proofread_srt(original)
            lines = [
                "편집한 SRT와 최초 OCR SRT의 차이입니다.",
                "교정본이 독립적인 정답임을 인증하지는 않습니다.",
                "",
                f"최초/교정 자막: {score.baseline_cues} / {score.corrected_cues}",
                f"시간으로 연결된 자막: {score.aligned_cues}",
                f"교정본 추가 / 원본에만 있음: "
                f"{score.corrected_only} / {score.baseline_only}",
                f"글자·문장부호·줄바꿈 완전 일치: "
                f"{score.exact_visible_cues} / {score.aligned_cues}",
                f"줄 수 동일: {score.exact_physical_lines} / {score.aligned_cues}",
                f"시작·종료 모두 0.25초 이내: "
                f"{score.precise_250ms_cues} / {score.aligned_cues}",
                f"문구와 시간 모두 일치: "
                f"{score.strict_joint_cues} / {score.aligned_cues}",
                f"교정본 기준 문자 오류율(CER): "
                f"{score.character_error_rate:.2%}",
                f"시작/종료 최대 차이: "
                f"{score.max_start_drift_seconds:.3f} / "
                f"{score.max_end_drift_seconds:.3f}초",
            ]
            if score.first_review_cue_numbers:
                lines.append(
                    "먼저 확인할 자막 번호(최대 16개): "
                    + ", ".join(str(x) for x in score.first_review_cue_numbers)
                )
            self.status_var.set("교정본과 최초 추출 자막 비교 완료")
            self._log("읽기 전용 교정 품질 비교 완료 · 문구·경로 저장 없음")
            messagebox.showinfo("교정 품질 비교 · 읽기 전용",
                                "\n".join(lines))
        except Exception as exc:
            self._log("교정 품질 비교 실패 · " + type(exc).__name__)
            messagebox.showerror(
                "교정 품질 비교",
                "원본과 '-proofread.srt' 교정 사본의 유효성을 확인하세요. "
                "두 파일은 변경하지 않았습니다.",
            )

    def _start_local_ocr_preview(self, *, compare_bright: bool = False):
        """Explicit local preflight; A/B compares the same source pixels."""
        if type(compare_bright) is not bool:
            raise TypeError("invalid preview comparison choice")
        if getattr(self, "_pending_close", False):
            return
        active_tab = getattr(self, "_tab_capture_thread", None)
        if active_tab is not None and active_tab.is_alive():
            return
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            from .local_ocr_preview import parse_preview_second
            if self.source_mode_var.get() != "local":
                raise ValueError("내 영상 1개 모드만 OCR 미리보기가 가능합니다.")
            source = Path(self.url_var.get().strip()).expanduser()
            if not source.is_file() or source.is_symlink():
                raise ValueError("선택한 로컬 동영상을 확인하세요.")
            second = parse_preview_second(self.preview_time_var.get())
            language = normalize_ocr_language(self.ocr_language_var.get())
            preview_backend = (
                "openvino"
                if bool(
                    self.fast_cpu_var.get()
                    if getattr(self, "fast_cpu_var", None) is not None
                    else False
                )
                else "onnxruntime"
            )
            if preview_backend == "openvino" and not openvino_runtime_available():
                raise ValueError("빠른 CPU(OpenVINO) 런타임이 없습니다.")
            focus = bool(self.focus_bottom_two_var.get()) if getattr(
                self, "focus_bottom_two_var", None
            ) is not None else False
            light = bool(self.light_caption_temporal_var.get()) if getattr(
                self, "light_caption_temporal_var", None
            ) is not None else False
            native = bool(self.native_vsf_var.get()) if getattr(
                self, "native_vsf_var", None
            ) is not None else False
            islands = bool(self.caption_islands_var.get()) if getattr(
                self, "caption_islands_var", None
            ) is not None else False
            if sum(bool(x) for x in (focus, light, islands, native)) > 1:
                raise ValueError("incompatible experimental preview choices")
            if compare_bright and (focus or islands or native):
                raise ValueError("A/B comparison excludes native and line-crop modes")
            selected_label = (
                self.local_ocr_region_var.get()
                if getattr(self, "local_ocr_region_var", None) is not None
                else "기본 · 하단 40%"
            )
            region_mode = _local_region_from_label(selected_label)
            from .tab_capture import roi_for_region
            roi_for_region(region_mode)
            if native and region_mode != "bottom40":
                raise ValueError("external native detector ROI unavailable")
            preview_mode = (
                "compare-bright" if compare_bright else
                "caption-islands" if islands else
                "light-caption" if light else
                "bottom-two-lines" if focus else "default"
            )
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("OCR 미리보기 입력", type(exc).__name__
                                 + " · 영상과 시각을 확인하세요.")
            return
        self._cancel_event.clear()
        self.progress_bar.stop()
        self.progress_bar.configure(mode="indeterminate", value=0)
        self.progress_bar.start(12)
        self.start_button.configure(state="disabled")
        if getattr(self, "review_button", None) is not None:
            self.review_button.configure(state="disabled")
        self.ocr_preview_button.configure(state="disabled")
        if getattr(self, "ocr_compare_button", None) is not None:
            self.ocr_compare_button.configure(state="disabled")
        if getattr(self, "subtitle_edit_button", None) is not None:
            self.subtitle_edit_button.configure(state="disabled")
        if getattr(self, "proofread_ass_button", None) is not None:
            self.proofread_ass_button.configure(state="disabled")
        if getattr(self, "corrected_quality_button", None) is not None:
            self.corrected_quality_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("원본 영상 한 프레임 OCR 확인 중")
        self._log("로컬 OCR 미리보기 시작 · 선택한 화면 필터 적용"
                  " · 프레임·원문·파일 저장 없음")
        self.worker = threading.Thread(
            target=self._local_ocr_preview_worker,
            args=(source, second, language) + (
                (preview_mode,) if preview_mode != "default" else ()
            ),
            kwargs={
                "region_mode": region_mode,
                "inference_engine": preview_backend,
            },
            daemon=True,
        )
        self.worker.start()

    def _local_ocr_preview_worker(
        self, source: Path, second: float, language: str,
        preview_mode: str = "default",
        *, region_mode: str = "bottom40",
        inference_engine: str = "onnxruntime",
    ) -> None:
        try:
            from .local_ocr_preview import inspect_local_ocr_preview
            from .extraction_progress import ExtractionCancelled
            def check():
                if self._cancel_event.is_set():
                    raise ExtractionCancelled()
            if preview_mode not in {
                "default", "bottom-two-lines", "light-caption",
                "caption-islands", "compare-bright",
            }:
                raise ValueError("unsupported preview mode")
            shared_adapter = [None]
            def make_ocr():
                models = bundled_model_dir()
                manifest = bundled_manifest_path()
                if not models.is_dir() or not manifest.is_file():
                    raise RuntimeError("verified original-language OCR models unavailable")
                if shared_adapter[0] is None:
                    shared_adapter[0] = self._make_rapidocr_adapter(
                        model_dir=str(models),
                        manifest_path=str(manifest),
                        language=language,
                        inference_engine=inference_engine,
                    )
                ocr = AdaptiveOCREngine(shared_adapter[0])
                if preview_mode == "bottom-two-lines":
                    from .ocr.line_focus import BottomTwoLineEngine
                    return BottomTwoLineEngine(ocr)
                if preview_mode == "light-caption":
                    from .ocr.light_text import LightCaptionEngine
                    return LightCaptionEngine(ocr)
                if preview_mode == "caption-islands":
                    from .ocr.caption_islands import CaptionIslandEngine
                    return CaptionIslandEngine(ocr)
                return ocr
            def make_comparison_ocr():
                from .ocr.light_text import LightCaptionEngine
                return LightCaptionEngine(make_ocr())
            result = inspect_local_ocr_preview(
                source, second, make_ocr, check_cancelled=check,
                cancel_requested=self._cancel_event.is_set,
                **({"preview_mode": preview_mode}
                   if preview_mode != "default" else {}),
                **({"region_mode": region_mode}
                   if region_mode != "bottom40" else {}),
                **({"compare_make_ocr": make_comparison_ocr}
                   if preview_mode == "compare-bright" else {}),
            )
            self.events.put(("local_ocr_preview_ready", result))
        except ExtractionCancelled:
            self.events.put(("cancelled", None))
        except Exception as exc:
            # A private frame, selected path, OCR text or FFmpeg stderr must
            # never enter the ordinary application logging event.
            self.events.put(("local_ocr_preview_error", type(exc).__name__))

    def _show_local_ocr_preview(self, result):
        """Render only on Tk's main thread; display text locally, never log it."""
        from PIL import Image, ImageTk
        from .local_ocr_preview import LocalOCRPreview
        if not isinstance(result, LocalOCRPreview):
            raise TypeError("invalid in-memory preview")
        window = tk.Toplevel(self.root)
        window.title("로컬 OCR 한 프레임 · 읽기 전용")
        window.geometry("785x530" if result.preview_mode == "compare-bright" else "740x440")
        pane = ttk.Frame(window, padding=12)
        pane.pack(fill="both", expand=True)
        # Frame source is BGR; Pillow expects RGB. Keep the image RAM-only.
        image = Image.fromarray(result.image[:, :, ::-1], "RGB")
        image.thumbnail((705, 245))
        photo = ImageTk.PhotoImage(image, master=window)
        picture = ttk.Label(pane, image=photo)
        picture.image = photo
        picture.pack(anchor="center", pady=(0, 7))
        ttk.Label(
            pane,
            text=(f"요청 {result.requested_second:.3f}초 · 실제 프레임 "
                  f"{result.actual_second:.3f}초 · OCR 신뢰도 "
                  f"{result.confidence:.2f} (정확도 보증 아님)"),
        ).pack(anchor="w")
        mode_labels = {
            "default": "기본 영상 OCR (선별·광학 보정 없음)",
            "bottom-two-lines": "실험: 하단 두 줄 판독 적용 (진짜 세 줄 자막 누락 가능)",
            "light-caption": "실험: 밝은 글씨 선별 적용 (시각 보정은 전체 추출에서만 수행)",
            "caption-islands": "실험: OCR 글자 위치로 대사 묶음 선별 (모호하면 원본 유지)",
            "compare-bright": "같은 영상 한 프레임의 기본/밝은 OCR 비교 · 기존 설정 불변",
        }
        ttk.Label(
            pane, text=mode_labels.get(result.preview_mode, "알 수 없는 OCR 모드"),
        ).pack(anchor="w", pady=(6, 0))
        region_help = {
            "bottom40": "하단 40% · 기존 기본",
            "bottom25": "하단 25% · 정상 윗줄 누락 가능",
            "lower65": "하단 65% · 장면 글자 오검출 가능",
            "lower80": "하단 80% · 장면 글자 오검출 가능",
        }
        ttk.Label(
            pane, text="실제 OCR 인식 범위: "
            + region_help.get(result.region_mode, "정보 없음"),
        ).pack(anchor="w", pady=(2, 0))
        if result.preview_mode == "compare-bright":
            if result.comparison_text is None or result.comparison_confidence is None:
                raise ValueError("incomplete comparison result")
            pair = ttk.Frame(pane)
            pair.pack(fill="both", expand=True, pady=(7, 0))
            for col, (label, value, confidence) in enumerate((
                ("기본 · 모든 글자", result.text, result.confidence),
                ("밝은 글자만 · 유색/어두운 대사 누락 위험",
                 result.comparison_text, result.comparison_confidence),
            )):
                column = ttk.Frame(pair)
                column.grid(row=0, column=col, sticky="nsew", padx=(0, 7))
                ttk.Label(column, text=f"{label} · 신뢰도 {confidence:.2f}").pack(anchor="w")
                words = tk.Text(column, height=7, wrap="word")
                words.pack(fill="both", expand=True)
                words.insert("1.0", value or "(감지된 글자 없음)")
                words.configure(state="disabled")
            pair.columnconfigure(0, weight=1)
            pair.columnconfigure(1, weight=1)
            pair.rowconfigure(0, weight=1)
        else:
            words = tk.Text(pane, height=6, wrap="word")
            words.pack(fill="both", expand=True, pady=(7, 0))
            words.insert(
                "1.0", result.text or (
                    "(이 프레임에서 선택한 필터를 통과한 글자 없음)"
                    if result.preview_mode != "default"
                    else "(이 프레임에서 인식한 자막 없음)"
                ),
            )
            words.configure(state="disabled")
        ttk.Label(pane, text="현재 원문 프레임 1장만 검사했습니다. "
                  "실제 전체 자막의 정확도는 보장되지 않으며 저장·전송하지 않습니다."
                  ).pack(anchor="w", pady=(8, 0))

    def _start_frame_review(self):
        """Start only when explicitly requested for a selected LOCAL video.

        Do not call the extraction preflight here: it rejects existing SRT
        files (which are precisely the input to the review workflow).
        """
        if getattr(self, "_pending_close", False):
            return
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            if self.source_mode_var.get() != "local":
                raise ValueError("원본 화면 검토는 내 영상 1개 모드에서만 가능합니다.")
            video = Path(self.url_var.get().strip()).expanduser()
            base = Path(self.out_var.get().strip()).expanduser()
            subtitle = base.with_suffix(".srt")
            if (not video.is_file() or video.is_symlink()
                    or not subtitle.is_file() or subtitle.is_symlink()):
                raise ValueError(
                    "영상과 이미 추출한 SRT 파일이 필요합니다. "
                    "기존 파일을 변경하지 않고 읽기만 합니다."
                )
            mode_label = self.review_mode_var.get()
            if mode_label not in {
                "경고 자막만", "추출 후 검토 권장",
                "모든 자막 순서대로", "반복 워터마크 후보",
                "연속 대사 분리 검토",
            }:
                raise ValueError("검토 범위를 선택하세요.")
            mode = (
                "all" if mode_label == "모든 자막 순서대로"
                else "temporal" if mode_label == "반복 워터마크 후보"
                else "split" if mode_label == "연속 대사 분리 검토"
                else "quality" if mode_label == "추출 후 검토 권장"
                else "flagged"
            )
            try:
                after = int(self.review_after_var.get().strip())
            except (ValueError, AttributeError) as exc:
                raise ValueError("이후 자막 번호는 0 이상의 정수여야 합니다.") from exc
            if after < 0 or after > 10000:
                raise ValueError("이후 자막 번호는 0~10000 범위여야 합니다.")
            lang = normalize_ocr_language(self.ocr_language_var.get())
            temporal_numbers: tuple[int, ...] = ()
            split_numbers: tuple[int, ...] = ()
            if mode == "quality":
                proof = getattr(self, "_quality_review_hints", None)
                if proof is None:
                    raise ValueError(
                        "이 영상의 추출 후 검토 근거가 없습니다. "
                        "새로 OCR 추출하거나 '경고 자막만'을 사용하세요."
                    )
                from hashlib import sha256
                original_video, original_srt, digest, size, mtime = proof
                current_stat = video.stat()
                if (
                    video != original_video or subtitle != original_srt
                    or current_stat.st_size != size
                    or current_stat.st_mtime_ns != mtime
                    or subtitle.stat().st_size > 8 * 1024 * 1024
                    or sha256(subtitle.read_bytes()).digest() != digest
                ):
                    raise ValueError(
                        "추출 후 원본 영상 또는 자막이 변경됐습니다. "
                        "이전 검토 번호는 사용할 수 없습니다."
                    )
            if mode == "temporal":
                hints = getattr(self, "_temporal_review_hints", None)
                if hints is None:
                    raise ValueError(
                        "이번 세션에 시간축 분석 기록이 없습니다. "
                        "검토 범위를 '반복 워터마크 후보'로 선택한 뒤 "
                        "내 영상 1개를 기본 OCR로 추출해야 합니다."
                    )
                from hashlib import sha256
                expected_video, expected_srt, digest, size, mtime, numbers = hints
                current_stat = video.stat()
                if (video != expected_video or subtitle != expected_srt
                    or current_stat.st_size != size
                    or current_stat.st_mtime_ns != mtime
                    or subtitle.stat().st_size > 8 * 1024 * 1024
                    or sha256(subtitle.read_bytes()).digest() != digest):
                    raise ValueError(
                        "추출 이후 영상 또는 자막 파일이 변경됐습니다. "
                        "이전 시간축 검토 후보는 재사용하지 않습니다."
                    )
                temporal_numbers = numbers
                if not temporal_numbers:
                    messagebox.showinfo(
                        "반복 글자 검토",
                        "확인된 반복 워터마크 후보가 없습니다. "
                        "실제 자막 정확도가 보증되는 것은 아닙니다."
                    )
                    return
            if mode == "split":
                proof = getattr(self, "_native_split_review_hints", None)
                if proof is None:
                    raise ValueError(
                        "이번 영상의 분리 검토 근거가 없습니다. "
                        "VideoSubFinder와 연속 대사 분리 옵션으로 다시 추출해야 합니다."
                    )
                from hashlib import sha256
                expected_video, expected_srt, digest, size, mtime, numbers = proof
                current_stat = video.stat()
                if (
                    video != expected_video or subtitle != expected_srt
                    or current_stat.st_size != size
                    or current_stat.st_mtime_ns != mtime
                    or subtitle.stat().st_size > 8 * 1024 * 1024
                    or sha256(subtitle.read_bytes()).digest() != digest
                ):
                    raise ValueError(
                        "추출 후 원본 영상이나 자막이 변경되어 "
                        "분리 검토 번호를 재사용할 수 없습니다."
                    )
                split_numbers = numbers
                if not split_numbers:
                    messagebox.showinfo(
                        "연속 대사 검토",
                        "이번 추출에서 확인된 분리 후보가 없습니다."
                    )
                    return
            # The user controls this output base, but the review gets a NEW
            # sibling directory. Repeat runs never overwrite earlier evidence.
            output = base.parent / (
                f"{base.name}.review-{mode}-after-{after:05d}"
            )
            if output.exists() or output.is_symlink():
                raise ValueError(
                    "이미 같은 이름의 검토 폴더가 있습니다. "
                    "기존 결과를 보존하려면 이후 자막 번호를 변경하세요."
                )
        except (OSError, ValueError) as exc:
            messagebox.showerror("원본 화면 검토 확인", str(exc))
            return

        if not messagebox.askyesno(
            "개인 영상 프레임 저장 승인",
            "선택한 로컬 영상에서 자막 앞·뒤 원본 화면을 최대 8개 자막"
            "(16장)까지 새 개인 폴더에 JPEG로 저장합니다.\n\n"
            "오프라인 HTML에는 이 SRT의 OCR 글자도 포함되며 검토 후"
            " 로컬 브라우저에서 자동으로 열립니다.\n\n"
            "프레임·자막에 개인정보가 있을 수 있습니다. 원본 영상·SRT는"
            " 변경하거나 외부 서버에 전송하지 않습니다. 계속하시겠습니까?",
        ):
            return
        self._cancel_event.clear()
        self._progress_stage = None
        self.progress_bar.stop()
        self.progress_bar.configure(mode="indeterminate", value=0)
        self.progress_bar.start(12)
        self.start_button.configure(state="disabled")
        self.review_button.configure(state="disabled")
        if getattr(self, "ocr_preview_button", None) is not None:
            self.ocr_preview_button.configure(state="disabled")
        if getattr(self, "ocr_compare_button", None) is not None:
            self.ocr_compare_button.configure(state="disabled")
        if getattr(self, "subtitle_edit_button", None) is not None:
            self.subtitle_edit_button.configure(state="disabled")
        if getattr(self, "proofread_ass_button", None) is not None:
            self.proofread_ass_button.configure(state="disabled")
        if getattr(self, "corrected_quality_button", None) is not None:
            self.corrected_quality_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("원본 자막 화면 검토 중")
        self._log("로컬 영상 검토 시작 · 선택한 자막 화면만 개인 폴더에 저장")
        self.worker = threading.Thread(
            target=self._review_worker,
            args=(video, subtitle, output, mode, after, lang)
                 + ((temporal_numbers,) if mode == "temporal"
                    else (split_numbers,) if mode == "split" else ()),
            daemon=True,
        )
        self.worker.start()

    def _review_worker(
        self, video: Path, subtitle: Path, output: Path,
        mode: str, after: int, language: str,
        temporal_cue_numbers: tuple[int, ...] | None = None,
    ):
        try:
            from .review_pack import create_pack

            def check_cancelled():
                if self._cancel_event.is_set():
                    raise ExtractionCancelled()

            result = create_pack(
                video, subtitle, output, mode=mode, max_cues=8,
                after_cue=after, language=language,
                check_cancelled=check_cancelled,
                **({"temporal_cue_numbers": temporal_cue_numbers}
                   if mode == "temporal" else
                   {"split_cue_numbers": temporal_cue_numbers}
                   if mode == "split" else {}),
            )
            # Publication is complete; do not claim cancellation afterwards.
            self.events.put(("review_done", (result, str(output))))
        except ExtractionCancelled:
            self.events.put(("cancelled", None))
        except Exception as exc:
            # Local file paths and actual frame contents never reach the
            # diagnostic log; errors are only classified by type.
            self.events.put(("error", {
                "summary": "원본 화면 검토에 실패했습니다. "
                           f"입력·저장 위치를 확인하세요 ({type(exc).__name__}).",
                "traceback": "HSE_REVIEW_FAILED=" + type(exc).__name__,
            }))

    def _worker(
        self,
        mode: str,
        url: str,
        target: str | None,
        out: Path,
        diagnostics_enabled: bool,
        languages: tuple[str, ...] | None = None,
        ocr_language: str = "ko",
        local_batch_paths: tuple[str, ...] = (),
    ):
        monitor = self._make_monitor()
        monitor_kwargs = {"monitor": monitor} if monitor is not None else {}
        try:
            if mode == "local":
                self._worker_local(
                    url, out, diagnostics_enabled, monitor,
                    ocr_language=ocr_language,
                    **({"selected_subtitle": target} if target else {}),
                    **({"region_mode": self._active_local_ocr_region}
                       if getattr(self, "_active_local_ocr_region", "bottom40") != "bottom40" else {}),
                    **({"native_vsf_opt_in": True}
                       if getattr(self, "_active_native_vsf_opt_in", False) else {}),
                    **({"native_split_opt_in": True}
                       if getattr(self, "_active_native_split_opt_in", False) else {}),
                    **({"seconv54_opt_in": True}
                       if getattr(self, "_active_seconv54_opt_in", False) else {}),
                    **({"focus_bottom_two_lines": True}
                       if getattr(self, "_active_focus_bottom_two", False) else {}),
                    **({"caption_islands": True}
                       if getattr(self, "_active_caption_islands", False) else {}),
                    **({"light_caption_temporal": True}
                       if getattr(self, "_active_light_caption_temporal", False) else {}),
                    **({"pixel_timing_only": True}
                       if getattr(self, "_active_pixel_timing_only", False) else {}),
                )
                return
            if mode == "local-batch":
                self._worker_local_batch(
                    url, out, monitor, ocr_language=ocr_language,
                    selected_paths=local_batch_paths,
                    **({"region_mode": self._active_local_ocr_region}
                       if getattr(self, "_active_local_ocr_region", "bottom40") != "bottom40" else {}),
                    **({"native_vsf_opt_in": True}
                       if getattr(self, "_active_native_vsf_opt_in", False) else {}),
                    **({"native_split_opt_in": True}
                       if getattr(self, "_active_native_split_opt_in", False) else {}),
                    **({"seconv54_opt_in": True}
                       if getattr(self, "_active_seconv54_opt_in", False) else {}),
                    **({"focus_bottom_two_lines": True}
                       if getattr(self, "_active_focus_bottom_two", False) else {}),
                    **({"caption_islands": True}
                       if getattr(self, "_active_caption_islands", False) else {}),
                    **({"light_caption_temporal": True}
                       if getattr(self, "_active_light_caption_temporal", False) else {}),
                    **({"pixel_timing_only": True}
                       if getattr(self, "_active_pixel_timing_only", False) else {}),
                )
                return
            if mode == "captions":
                self._worker_public_captions(
                    url, target, out, languages or ("ko",), monitor,
                )
                return
            self.events.put(("log", "1/4 사이트 회차를 확인하고 최적 추출 경로를 선택합니다."))
            client = AutoEpSiteClient()

            diagnostics = (
                DiagnosticRecorder(str(out) + ".diagnostics")
                if diagnostics_enabled
                else None
            )

            def prepare_ocr():
                self.events.put(("log", f"별도 자막이 없어 {ocr_language} OCR 모델을 준비합니다."))
                model_dir = bundled_model_dir()
                if not model_dir.is_dir():
                    raise RuntimeError(
                        "이 회차는 하드서브라 OCR 모델이 필요하지만 모델 번들이 없습니다."
                    )
                return AdaptiveOCREngine(
                    self._make_rapidocr_adapter(
                        model_dir=str(model_dir),
                        manifest_path=str(bundled_manifest_path()),
                        language=ocr_language,
                    )
                )
            self.events.put(("log", "2/4 별도 자막을 우선 확인하고, 없으면 직접 OCR 또는 브라우저 렌더 OCR을 사용합니다."))

            if target:
                self.events.put(("log", "3/4 원본 스트림의 visual sync 지원 여부를 먼저 확인한 뒤 자막을 추출합니다."))
                result = extract_and_sync_epsite(
                    url,
                    target,
                    out,
                    ocr=None,
                    ocr_factory=prepare_ocr,
                    ocr_language=ocr_language,
                    client=client,
                    diagnostics=diagnostics,
                    streaming=True,
                    **monitor_kwargs,
                )
                final_srt = result.synced.srt_path
                final_ass = result.synced.ass_path
                sync_report = result.synced.report_path
                self.events.put(("log", f"추출 경로: {result.extracted.mode}"))
                self.events.put(("log", f"싱크 구간: {len(result.synced.sync.segments)}개"))
                self.events.put(("log", f"sync report: {sync_report}"))
            else:
                self.events.put(("log", "3/4 사이트 기준 자막을 생성합니다."))
                result = extract_epsite(
                    url,
                    out,
                    ocr=None,
                    ocr_factory=prepare_ocr,
                    ocr_language=ocr_language,
                    client=client,
                    diagnostics=diagnostics,
                    streaming=True,
                    **monitor_kwargs,
                )
                final_srt = result.srt_path
                final_ass = result.ass_path
                self.events.put(("log", f"추출 경로: {result.mode}"))

            if diagnostics is not None:
                diag = diagnostics.write(
                    extra={"source": url, "target_video": target}
                )
                self.events.put(("log", f"diagnostics: {diag}"))

            self.events.put(("log", "4/4 완료"))
            self.events.put(("log", f"SRT: {final_srt}"))
            self.events.put(("log", f"ASS: {final_ass}"))
            self.events.put(("done", (str(final_srt), str(final_ass))))
        except ExtractionCancelled:
            self.events.put(("cancelled", None))
        except Exception as exc:
            self.events.put(
                (
                    "error",
                    {
                        "summary": _user_error_summary(exc),
                        "traceback": _safe_gui_error_text(traceback.format_exc()),
                    },
                )
            )

    def _worker_public_captions(
        self, source: str, url_file: str | None, out: Path,
        languages: tuple[str, ...], monitor: ExtractionMonitor | None = None,
    ) -> None:
        if monitor is not None:
            monitor.check_cancelled()
        urls = load_episode_urls([source], url_file)
        video_ocr = bool(getattr(self, "_requested_video_ocr_enabled", False))
        if video_ocr:
            self.events.put(("log", "웹 자막 우선 수집 · 공개 자막이 실제 없을 때만 임시 영상 OCR"))
        else:
            self.events.put(("log", "웹 자막 전용 모드 · 영상 다운로드 및 OCR 사용 안 함"))
        self.events.put((
            "log", "공개 영상의 언어별 자막을 우선 확인하며 인증·DRM 제한은 우회하지 않습니다."
        ))
        route = getattr(self, "_caption_proxy_route", CaptionProxyRoute())
        self.events.put(("log", f"웹 자막 연결: {route.mode} (프록시 주소는 기록하지 않음)"))
        collector = PublicCaptionCollector(proxy_route=route)
        downloaded = 0
        needs_ocr = 0
        unavailable = 0
        existing_preserved = 0
        existing_verified = 0
        for index, episode in enumerate(urls, 1):
            if monitor is not None:
                monitor.check_cancelled()
            self.events.put(("caption_progress", (index - 1, len(urls))))
            try:
                results = collector.collect(
                    episode, out, languages=languages, skip_existing=True,
                    **({"check_cancelled": monitor.check_cancelled} if monitor else {}),
                )
                if video_ocr and any(item.status == "needs_ocr" for item in results):
                    self.events.put((
                        "log", f"회차 {index}: 공개 자막이 없는 언어의 영상 OCR 경로 확인"
                    ))
                    try:
                        results = resolve_missing_public_ocr(
                            episode, out, results, proxy_route=route,
                            monitor=monitor, skip_existing=True,
                        )
                    except PublicVideoOCRError:
                        # Failed video transport/probe must not erase an
                        # already-successful manual/automatic caption result.
                        # Count only the still-missing languages as failures.
                        results = [
                            replace(item, status="ocr_unavailable")
                            if item.status == "needs_ocr" else item
                            for item in results
                        ]
            except FileExistsError:
                unavailable += len(languages)
                self.events.put(("log", f"회차 {index}: 기존 자막 보존 · 파일 변경 없음"))
                self.events.put(("caption_progress", (index, len(urls))))
                continue
            except CaptionAccessError:
                unavailable += len(languages)
                failure = (
                    "프록시 필수 경로 접근 불가 · DIRECT 미시도"
                    if route.mode == "required" else
                    "프록시 재시도 접근 불가" if route.mode == "fallback"
                    else "공개 자막 접근 불가"
                )
                self.events.put(("log", f"회차 {index}: {failure}"))
                self.events.put(("caption_progress", (index, len(urls))))
                continue
            except (OSError, ValueError, PublicVideoOCRError):
                # The downloader/probe can reject an untrusted/oversized
                # video after caption inspection. Treat this one episode as
                # inaccessible, not as a crash that discards the rest of a
                # 50-episode batch. Do not disclose provider error strings.
                unavailable += len(languages)
                self.events.put(("log", f"회차 {index}: 공개 자막 또는 영상 검증 실패"))
                self.events.put(("caption_progress", (index, len(urls))))
                continue
            for result in results:
                if result.status in {"downloaded", "hard_ocr"}:
                    downloaded += 1
                elif result.status == "needs_ocr":
                    needs_ocr += 1
                elif result.status == "existing_verified":
                    existing_verified += 1
                elif result.status == "existing_preserved":
                    existing_preserved += 1
                else:
                    unavailable += 1
                # Never put episode URLs, provider IDs or subtitle words into logs.
                self.events.put((
                    "log",
                    f"회차 {index} · {result.requested_language} · {result.status}"
                    f" · {result.cue_count}개 자막 · 경로 {result.connection_route}",
                ))
            self.events.put(("caption_progress", (index, len(urls))))
        # Keep the original terminal payload compatible with old GUI users;
        # audited results are reported as a separate, ordered event.
        self.events.put(("caption_verified", existing_verified))
        self.events.put(("captions_done", (
            downloaded, needs_ocr, unavailable, existing_preserved, str(out)
        )))

    def _emit_local_ocr_quality(self, video: Path, srt: Path, language: str) -> None:
        # After successful extraction, read-only advisory; never modify SRT/ASS.
        try:
            from .local_ocr_quality_signals import local_ocr_quality_signals
            result = local_ocr_quality_signals(video, srt, language=language)
        except (OSError, RuntimeError, ValueError, TypeError):
            self.events.put(("ocr_quality_unavailable", None))
        else:
            samples = tuple(result.get("review_cue_numbers", ()))[:8]
            self.events.put(("ocr_quality_signal", int(result["review_signal_count"])))
            if samples:
                self.events.put(("ocr_quality_cue_numbers", samples))
                # Bind the precise cue index set to this video's metadata
                # and to the actual exported subtitle bytes. This is not a
                # cryptographic hash of the entire original movie.
                try:
                    from hashlib import sha256
                    video_stat = Path(video).stat()
                    subtitle_path = Path(srt)
                    if (Path(video).is_symlink() or subtitle_path.is_symlink()
                            or not subtitle_path.is_file()
                            or not 0 < subtitle_path.stat().st_size <= 8 * 1024 * 1024):
                        raise ValueError("invalid_review_identity")
                    digest = sha256(subtitle_path.read_bytes()).digest()
                except (OSError, ValueError, TypeError):
                    self.events.put(("ocr_quality_bound_unavailable", None))
                else:
                    self.events.put(("ocr_quality_bound_hints", (
                        Path(video), subtitle_path, digest,
                        video_stat.st_size, video_stat.st_mtime_ns,
                    )))

    def _worker_local_batch(
        self, file_list: str, out: Path,
        monitor: ExtractionMonitor | None = None, *,
        ocr_language: str = "ko",
        selected_paths: tuple[str, ...] = (),
        native_vsf_opt_in: bool = False,
        native_split_opt_in: bool = False,
        seconv54_opt_in: bool = False,
        focus_bottom_two_lines: bool = False,
        caption_islands: bool = False,
        light_caption_temporal: bool = False,
        pixel_timing_only: bool = False,
        region_mode: str = "bottom40",
    ) -> None:
        items = load_local_batch(
            selected_paths, None if selected_paths else file_list, out,
        )
        total = len(items)
        self.events.put((
            "log",
            f"로컬 영상 {total}개 · {ocr_language} 원본 내장/별도 자막 우선"
            " · 미발견 회차만 임시 OCR · 기존 결과 덮어쓰기 금지",
        ))
        if pixel_timing_only:
            self.events.put(("log",
                "실험: 기본 OCR 문구·개수 보존, 픽셀 시각만 선택 보정"
                " · 기존 자막 보존 · CPU 재디코딩 · 정확도 별도 확인 필요"
            ))
        if light_caption_temporal:
            self.events.put(("log",
                "실험: 밝은 자막 전용 광학 시각 보정 · 회차별 별도 근거"
                " · 원본 자막 우선 · 추가 모델 호출 없음"
                " · CPU 재디코딩 · 어두운/일부 색 글자 유실 가능"
            ))
        self.events.put(("caption_progress", (0, total)))

        def new_ocr():
            model_dir = bundled_model_dir()
            if not model_dir.is_dir():
                raise RuntimeError("로컬 일괄 OCR 모델이 없습니다.")
            return AdaptiveOCREngine(
                self._make_rapidocr_adapter(
                    model_dir=str(model_dir),
                    manifest_path=str(bundled_manifest_path()),
                    language=ocr_language,
                )
            )

        def on_item(item):
            # Show only sanitized status and counts; the source path/name,
            # captions, original subtitle bytes and private file details are
            # never included in logs or queue events.
            missed = (
                f" · 미인식 구간 {item.missed_events}개"
                if item.mode in {
                    "hard_ocr_native_timing",
                    "hard_ocr_native_timing_partial",
                    "local-embedded-image-ocr-partial",
                } else ""
            )
            self.events.put((
                "log", f"영상 {item.index}/{total} · {item.mode}"
                f" · 자막 {item.cues}개" + missed,
            ))
            short = getattr(item, "short_frame_rescue_attempted", 0)
            deferred = getattr(item, "short_frame_rescue_deferred", 0)
            errors = getattr(item, "short_frame_rescue_errors", 0)
            if short or deferred or errors:
                self.events.put(("log",
                    f"영상 {item.index}번 · 짧은 자막 추가 프레임 확인 {short}개"
                    f" · 복원 {getattr(item, 'short_frame_rescue_recognized', 0)}개"
                    f" · 처리 상한 보류 {deferred}개"
                    f" · 확인 실패 {errors}개"
                ))
            split_cues = tuple(getattr(item, "split_cue_numbers", ()))[:8]
            if split_cues:
                self.events.put(("log",
                    f"영상 {item.index}번 · 연속 대사 분리 검토 자막쌍 시작 번호: "
                    + ", ".join(str(int(n)) for n in split_cues)
                    + " · 각 번호와 바로 다음 자막을 원본 화면과 비교"
                ))
            windows = tuple(getattr(item, "missed_windows", ()))
            if windows:
                from .timed_ocr import format_missed_windows
                self.events.put(("log",
                    f"영상 {item.index}번 · OCR 미인식 후보 시각 (최대 8개): "
                    + format_missed_windows(windows)
                ))
            if (light_caption_temporal or pixel_timing_only) and item.mode == "hard_ocr":
                self.events.put(("log",
                    f"영상 {item.index}번 · 광학 시각 보정 후보"
                    f" {item.optical_timing_candidates}개"
                    f" · 보정 {item.optical_timing_refined}개"
                    " (정답 정확도 인증 아님)"
                ))
                if item.optical_witness_stride > 1:
                    self.events.put(("log",
                        f"영상 {item.index}번 · 장편 광학 근거 예산 조정"
                        " (서로 다른 대사 우선·영상 전체 분산,"
                        " 일부 초단 대사 시간 보정은 제한될 수 있음)"
                    ))
            self.events.put(("caption_progress", (item.index, total)))

        def new_native_tool():
            # A single consent at Start authorizes the optional free helper
            # ONLY for episodes with no usable source subtitle track. The
            # batch backend caches this result across all such episodes.
            from .native_vsf_assets import ensure_official_vsf
            release = getattr(self, "_active_native_vsf_release", "v6.10.5")
            return ensure_official_vsf(
                consent=True,
                check_cancelled=monitor.check_cancelled if monitor else None,
                **({"release_version": release} if release != "v6.10.5" else {}),
            )

        seconv_candidate = None
        def new_seconv():
            nonlocal seconv_candidate
            if seconv_candidate is None:
                from .managed_seconv import ensure_official_seconv_54
                seconv_candidate = ensure_official_seconv_54(
                    consent=True,
                    check_cancelled=monitor.check_cancelled if monitor else None,
                )
            return seconv_candidate

        results = run_local_batch(
            items, language=ocr_language, ocr_factory=new_ocr,
            check_cancelled=monitor.check_cancelled if monitor else None,
            monitor=monitor, on_result=on_item,
            **({"region_mode": region_mode} if region_mode != "bottom40" else {}),
            **({"focus_bottom_two_lines": True}
               if focus_bottom_two_lines else {}),
            **({"caption_islands": True}
               if caption_islands else {}),
            **({"light_caption_temporal": True}
               if light_caption_temporal else {}),
            **({"pixel_timing_only": True}
               if pixel_timing_only else {}),
            **({"native_timing_factory": new_native_tool}
               if native_vsf_opt_in else {}),
            **({"native_split_opt_in": True}
               if native_split_opt_in else {}),
            **({"image_converter_factory": new_seconv}
               if seconv54_opt_in else {}),
        )
        # Read-only post-export quality review only for newly OCR-produced
        # episodes, never original source subtitles or already existing files.
        audited = flagged = review_unavailable = 0
        ocr_modes = {
            "local-embedded-image-ocr", "local-embedded-image-ocr-partial",
            "hard_ocr", "hard_ocr_native_timing", "hard_ocr_native_timing_partial",
        }
        from .local_ocr_quality_signals import local_ocr_quality_signals
        for item, record in zip(items, results):
            if record.mode not in ocr_modes:
                continue
            if monitor is not None:
                monitor.check_cancelled()
            try:
                signals = local_ocr_quality_signals(
                    item.source, item.out_base.with_suffix(".srt"),
                    language=ocr_language,
                )
            except (OSError, RuntimeError, ValueError, TypeError):
                review_unavailable += 1
            else:
                audited += 1
                if signals["review_signal_count"]:
                    flagged += 1
                    self.events.put(("batch_ocr_quality_item", (
                        record.index, int(signals["review_signal_count"]),
                        tuple(signals.get("review_cue_numbers", ()))[:8],
                    )))
        self.events.put(("batch_ocr_quality",
                         (audited, flagged, review_unavailable)))
        completed = sum(
            r.mode in {"local-embedded-text", "local-sidecar-text", "local-embedded-image-ocr", "hard_ocr", "hard_ocr_native_timing"}
            for r in results
        )
        incomplete = sum(
            r.mode in {
                "hard_ocr_native_timing_partial",
                "local-embedded-image-ocr-partial",
            } for r in results
        )
        verified = sum(r.mode == "existing_verified" for r in results)
        unavailable = sum(r.mode == "failed" for r in results)
        preserved = sum(r.mode == "existing_preserved" for r in results)
        # Structural verification of completed original text subtitles is
        # distinct from both newly generated files and unknown OCR outputs.
        self.events.put(("caption_verified", verified))
        self.events.put(("captions_done", (
            completed, incomplete, unavailable, preserved, str(out),
        )))

    def _worker_local(
        self, source: str, out: Path, diagnostics_enabled: bool,
        monitor: ExtractionMonitor | None = None,
        *,
        ocr_language: str = "ko",
        selected_subtitle: str | None = None,
        native_vsf_opt_in: bool = False,
        native_split_opt_in: bool = False,
        seconv54_opt_in: bool = False,
        focus_bottom_two_lines: bool = False,
        caption_islands: bool = False,
        light_caption_temporal: bool = False,
        pixel_timing_only: bool = False,
        region_mode: str = "bottom40",
    ):
        from .tab_capture import roi_for_region
        crop = roi_for_region(region_mode)
        if region_mode != "bottom40" and native_vsf_opt_in:
            raise ValueError("independent native detector cannot accept ROI")
        if type(caption_islands) is not bool or (
            caption_islands and (
                native_vsf_opt_in or focus_bottom_two_lines or light_caption_temporal
            )
        ):
            raise ValueError("caption-island mode conflicts with other OCR styles")
        if type(light_caption_temporal) is not bool or (
            light_caption_temporal and (native_vsf_opt_in or focus_bottom_two_lines)
        ):
            raise ValueError("light-caption optical timing incompatible with other modes")
        if type(pixel_timing_only) is not bool or (
            pixel_timing_only and (
                light_caption_temporal or caption_islands
                or native_vsf_opt_in or focus_bottom_two_lines
            )
        ):
            raise ValueError("pixel-only optical timing incompatible with other OCR modes")
        if type(focus_bottom_two_lines) is not bool or (
            focus_bottom_two_lines and native_vsf_opt_in
        ):
            raise ValueError("bottom-two-line focus incompatible with native timing")
        self.events.put((
            "log", f"내 영상의 {ocr_language} 내장·별도 원본 자막을 OCR보다 먼저 확인합니다."
        ))
        if monitor is not None:
            monitor.report("resolving")
        cancel_kwargs = (
            {"check_cancelled": monitor.check_cancelled} if monitor else {}
        )
        if selected_subtitle:
            soft = acquire_local_sidecar_text(
                source, out, language=ocr_language,
                selected_path=selected_subtitle, **cancel_kwargs,
            )
        else:
            soft = acquire_local_embedded_text(
                source, out, language=ocr_language, **cancel_kwargs,
            )
            if soft is None:
                soft = acquire_local_sidecar_text(
                    source, out, language=ocr_language, **cancel_kwargs,
                )
            if soft is None:
                seconv_factory = None
                if seconv54_opt_in:
                    from .managed_seconv import ensure_official_seconv_54
                    seconv_factory = lambda: ensure_official_seconv_54(
                        consent=True,
                        check_cancelled=monitor.check_cancelled if monitor else None,
                    )
                soft = acquire_local_embedded_image(
                    source, out, language=ocr_language,
                    ocr_factory=lambda: AdaptiveOCREngine(
                        self._make_rapidocr_adapter(
                            model_dir=str(bundled_model_dir()),
                            manifest_path=str(bundled_manifest_path()),
                            language=ocr_language,
                        )
                    ),
                    **({"converter_factory": seconv_factory}
                       if seconv_factory is not None else {}),
                    **cancel_kwargs,
                )
        if soft is not None:
            self.events.put((
                "log", (
                    f"{soft.mode} · " +
                    (f"별도 이미지 트랙 OCR {soft.cue_count}개"
                     if soft.mode == "local-embedded-image-ocr"
                     else f"원본 텍스트 자막 {soft.cue_count}개 · OCR 모델 불필요")
                )
            ))
            self.events.put(("log", f"원본 자막 출처 기록: {soft.provenance_path}"))
            if monitor is not None:
                monitor.finish(soft.cue_count)
            source_missed = getattr(soft, "missed_events", 0)
            if soft.mode == "local-embedded-image-ocr":
                self._emit_local_ocr_quality(Path(source), Path(soft.srt_path), ocr_language)
            if soft.mode == "local-embedded-image-ocr" and source_missed:
                self.events.put((
                    "log",
                    f"원본 이미지 자막 {source_missed}개 구간은 OCR로 읽지 못했습니다."
                    " 일부 자막 파일을 보존하며 완료로 표시하지 않습니다.",
                ))
                self.events.put(("partial_done", (
                    str(soft.srt_path), str(soft.ass_path), int(source_missed),
                )))
            else:
                self.events.put(("done", (str(soft.srt_path), str(soft.ass_path))))
            return
        self.events.put((
            "log", f"일치하는 내장·별도 텍스트 자막이 없어 {ocr_language} 하드자막 OCR을 시작합니다."
        ))
        if native_vsf_opt_in:
            if monitor is not None:
                monitor.check_cancelled()
            model_dir = bundled_model_dir()
            if not model_dir.is_dir():
                raise RuntimeError("로컬 영상 OCR에 필요한 모델 번들이 없습니다.")
            self.events.put(("log",
                "실험용 공식 VideoSubFinder 구간 감지 → HSE 원문 OCR 진행"
                " (공식 SHA 검증, 사용자 전용 캐시, 영상 업로드 없음)"
            ))
            from .native_vsf_assets import ensure_official_vsf
            from .native_vsf_timing import extract_with_native_vsf
            cancel = monitor.check_cancelled if monitor is not None else None
            release = getattr(self, "_active_native_vsf_release", "v6.10.5")
            executable, digest = ensure_official_vsf(
                consent=True, check_cancelled=cancel,
                **({"release_version": release} if release != "v6.10.5" else {}),
            )
            measured = extract_with_native_vsf(
                source, out, executable=executable,
                pinned_executable_sha256=digest,
                ocr_factory=lambda: AdaptiveOCREngine(
                    self._make_rapidocr_adapter(
                        model_dir=str(model_dir),
                        manifest_path=str(bundled_manifest_path()),
                        language=ocr_language,
                    )
                ),
                check_cancelled=cancel,
                **({"split_stable_text": True} if native_split_opt_in else {}),
            ).report
            if monitor is not None:
                monitor.finish(measured.recognized_intervals)
            self._emit_local_ocr_quality(Path(source), Path(measured.srt_path), ocr_language)
            self.events.put(("log",
                f"실험 탐지 {measured.detected_intervals}개"
                f" · OCR 성공 {measured.recognized_intervals}개"
                f" · 미인식 {measured.missed_intervals}개"
                f" · 출력 {getattr(measured, 'exported_cues', measured.recognized_intervals)}개"
                f" · 연속 대사 분리 {getattr(measured, 'split_source_intervals', 0)}구간"
                f" · OCR 호출 {measured.ocr_calls}회"
            ))
            short = getattr(measured, "short_frame_rescue_attempted", 0)
            deferred = getattr(measured, "short_frame_rescue_deferred", 0)
            errors = getattr(measured, "short_frame_rescue_errors", 0)
            if short or deferred or errors:
                self.events.put(("log",
                    f"짧은 자막 추가 프레임 확인 {short}개"
                    f" · 복원 {getattr(measured, 'short_frame_rescue_recognized', 0)}개"
                    f" · 처리 상한 보류 {deferred}개"
                    f" · 확인 실패 {errors}개"
                    " · 실제 글자가 없어 복원되지 않았을 수도 있음"
                ))
            split_cues = (
                tuple(getattr(measured, "split_cue_numbers", ()))[:8]
                if native_split_opt_in else ()
            )
            if split_cues:
                # Evidence is valid only for this source's newly exported SRT.
                # Keep user paths, frames and subtitle words in RAM, not logs.
                try:
                    from hashlib import sha256
                    video_file = Path(source)
                    srt_file = Path(measured.srt_path)
                    stat = video_file.stat()
                    if (
                        video_file.is_symlink() or srt_file.is_symlink()
                        or not srt_file.is_file()
                        or not 0 < srt_file.stat().st_size <= 8 * 1024 * 1024
                    ):
                        raise ValueError("invalid_native_split_review_source")
                    self.events.put(("native_split_review_ready", (
                        video_file, srt_file, sha256(srt_file.read_bytes()).digest(),
                        stat.st_size, stat.st_mtime_ns, split_cues,
                    )))
                except (OSError, ValueError, TypeError):
                    self.events.put(("log",
                        "분리된 자막 번호는 표시하지만 원본 검토 근거 인증은 보류됐습니다."
                    ))
                self.events.put(("log",
                    "분리된 자막쌍 시작 번호 (최대 8개): "
                    + ", ".join(str(int(n)) for n in split_cues)
                    + " · 해당 번호와 다음 번호를 원본 화면 검토에서 확인"
                ))
            windows = tuple(getattr(measured, "missed_windows", ()))
            if windows:
                from .timed_ocr import format_missed_windows
                self.events.put(("log",
                    "OCR 미인식 후보 시각 (최대 8개): "
                    + format_missed_windows(windows)
                    + " · 검출기 오탐 가능, 원본 확인 필요"
                ))
            self.events.put((
                "partial_done" if measured.missed_intervals else "done",
                (str(measured.srt_path), str(measured.ass_path),
                 measured.missed_intervals)
                if measured.missed_intervals else
                (str(measured.srt_path), str(measured.ass_path)),
            ))
            return
        if monitor is not None:
            monitor.check_cancelled()
        model_dir = bundled_model_dir()
        if not model_dir.is_dir():
            raise RuntimeError("로컬 영상 OCR에 필요한 모델 번들이 없습니다.")
        diagnostics = DiagnosticRecorder(str(out) + ".diagnostics") if diagnostics_enabled else None
        temporal_audit = None
        combined_diagnostics = diagnostics
        if getattr(self, "_active_temporal_review_opt_in", False):
            from .temporal_review import (
                TemporalReviewRecorder, ChainedSegmentAudits)
            temporal_audit = TemporalReviewRecorder()
            combined_diagnostics = ChainedSegmentAudits(
                diagnostics, temporal_audit)
        ocr = AdaptiveOCREngine(self._make_rapidocr_adapter(
            model_dir=str(model_dir), manifest_path=str(bundled_manifest_path()),
            language=ocr_language,
        ))
        light = None
        if caption_islands:
            from .ocr.caption_islands import CaptionIslandEngine
            ocr = CaptionIslandEngine(ocr)
            self.events.put(("log",
                "실험: OCR 글자 위치로 대사 묶음 선별"
                " · 모호한 묶음은 보존 · 기본 자막 우선 · 정확도 검토 필요"
            ))
        if light_caption_temporal or pixel_timing_only:
            from functools import partial
            from .ocr.light_temporal import LightTemporalWitnessEngine
            if light_caption_temporal:
                from .ocr.light_text import LightCaptionEngine
                ocr = LightCaptionEngine(ocr)
            # Pixel-only preserves the exact default OCR voting and cue set.
            # The pipeline contract rejects any refiner that changes text/count.
            light = LightTemporalWitnessEngine(ocr)
            ocr = light
            if pixel_timing_only:
                self.events.put(("log",
                    "실험: 원문·자막 개수 보존 + 픽셀 표시시각만 보정"
                    " · 정확도 인증 아님 · 추가 OCR 모델 호출 없음"
                ))
            else:
                self.events.put(("log",
                    "실험: 밝은 글씨·어두운 배경 자막 선별 후 영상 픽셀로"
                    " 표시시각 보정 · 어두운/색 자막 유실 가능 · CPU 재검사"
                    " · 추가 모델 호출 없음"
                ))
        if monitor is not None:
            monitor.check_cancelled()
        result = extract_local_video(
            source, ocr, out, diagnostics=combined_diagnostics,
            streaming=True,
            **({"roi": crop} if region_mode != "bottom40" else {}),
            **({"monitor": monitor} if monitor is not None else {}),
            **({"focus_bottom_two_lines": True}
               if focus_bottom_two_lines else {}),
            **({"timing_refiner": partial(
                light.refine_timings,
                check_cancelled=monitor.check_cancelled if monitor else None,
                cancel_requested=monitor.is_cancelled if monitor else None,
            )} if light is not None else {}),
        )
        if light is not None:
            self.events.put(("log",
                f"광학 시각 보정 후보 {light.candidate_cues}개"
                f" · 수정 {light.refined_cues}개 · 보류 {light.skipped_cues}개"
                " (정답 정확도 인증 아님)"
            ))
            if light.witness_sample_stride > 1:
                self.events.put(("log",
                    "장편 광학 근거 예산 조정: 서로 다른 대사 우선"
                    " · 영상 전체 시점 분산 보존"
                    " (일부 짧은 대사 보정은 제한될 수 있음)"
                ))
        if diagnostics is not None:
            diag = diagnostics.write(extra={"mode": "local-hard-ocr", "source": source})
            self.events.put(("log", f"diagnostics: {diag}"))
        if result.cue_count == 0:
            raise RuntimeError("인식된 자막이 없습니다. 선택한 OCR 언어와 영상 자막을 확인하세요.")
        if temporal_audit is not None:
            try:
                from hashlib import sha256
                from .frame_source import probe_video
                from .quality_audit_io import _read_subtitles
                video_file = Path(source)
                srt_file = Path(result.srt_path)
                cues = _read_subtitles(srt_file)
                duration = probe_video(str(video_file)).duration
                numbers = temporal_audit.cue_numbers(cues, duration)
                stat = video_file.stat()
                evidence = (
                    video_file, srt_file, sha256(srt_file.read_bytes()).digest(),
                    stat.st_size, stat.st_mtime_ns, numbers,
                )
                self.events.put(("temporal_review_ready", evidence))
                self.events.put(("log",
                    f"시간축 반복 글자 의심 자막 {len(numbers)}개 · "
                    "검토 버튼에서 원본 프레임 확인 가능 · 자동 삭제 없음"))
            except (OSError, ValueError, RuntimeError, TypeError):
                self.events.put(("log",
                    "시간축 검토: 증거 부족·영상 길이 제한·관측량 초과 중 하나로 "
                    "후보 판정을 보류했습니다. 원본 자막은 보존했습니다."))
        self._emit_local_ocr_quality(Path(source), Path(result.srt_path), ocr_language)
        self.events.put(("log", f"인식한 자막: {result.cue_count}개"))
        self.events.put(("log", f"SRT: {result.srt_path}"))
        self.events.put(("log", f"ASS: {result.ass_path}"))
        self.events.put(("done", (str(result.srt_path), str(result.ass_path))))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind in {
                    "done", "captions_done", "partial_done",
                    "cancelled", "error", "review_done",
                    "local_ocr_preview_ready", "local_ocr_preview_error",
                } and getattr(self, "review_button", None) is not None:
                    self.review_button.configure(
                        state="normal"
                        if self.source_mode_var.get() == "local"
                        else "disabled"
                    )
                if kind in {
                    "done", "captions_done", "partial_done",
                    "cancelled", "error", "review_done",
                    "local_ocr_preview_ready", "local_ocr_preview_error",
                } and getattr(self, "ocr_preview_button", None) is not None:
                    self.ocr_preview_button.configure(
                        state="normal" if self.source_mode_var.get() == "local"
                        and not getattr(self, "_pending_close", False)
                        else "disabled"
                    )
                    if getattr(self, "ocr_compare_button", None) is not None:
                        self.ocr_compare_button.configure(
                            state="normal" if self.source_mode_var.get() == "local"
                            and not getattr(self, "_pending_close", False)
                            else "disabled"
                        )
                if kind in {
                    "done", "captions_done", "partial_done",
                    "cancelled", "error", "review_done",
                    "local_ocr_preview_ready", "local_ocr_preview_error",
                } and getattr(self, "subtitle_edit_button", None) is not None:
                    self.subtitle_edit_button.configure(
                        state="normal" if self.source_mode_var.get() == "local"
                        and not getattr(self, "_pending_close", False)
                        else "disabled"
                    )
                if kind in {
                    "done", "captions_done", "partial_done",
                    "cancelled", "error", "review_done",
                    "local_ocr_preview_ready", "local_ocr_preview_error",
                } and getattr(self, "proofread_ass_button", None) is not None:
                    self.proofread_ass_button.configure(
                        state="normal" if self.source_mode_var.get() == "local"
                        and not getattr(self, "_pending_close", False)
                        else "disabled"
                    )
                if kind in {
                    "done", "captions_done", "partial_done",
                    "cancelled", "error", "review_done",
                    "local_ocr_preview_ready", "local_ocr_preview_error",
                } and getattr(self, "corrected_quality_button", None) is not None:
                    self.corrected_quality_button.configure(
                        state="normal" if self.source_mode_var.get() == "local"
                        and not getattr(self, "_pending_close", False)
                        else "disabled"
                    )
                if getattr(self, "_pending_close", False):
                    # Drain queued events without opening modal dialogs while
                    # the application is already closing.
                    continue
                if kind == "ocr_quality_signal":
                    self._last_local_ocr_quality_signals = int(payload)
                    if int(payload):
                        self._log(
                            f"추출 자막 검토 신호 {int(payload)}건 · "
                            "원본 화면 검토 (수동)에서 확인 가능 · 자동 수정 없음"
                        )
                    continue
                if kind == "ocr_quality_cue_numbers":
                    self._last_local_quality_samples = tuple(payload)[:8]
                    if self._last_local_quality_samples:
                        listed = ", ".join(str(int(n)) for n in self._last_local_quality_samples)
                        self._log("원본 화면 검토 참고 · 의심 자막 번호 " + listed)
                    continue
                if kind == "batch_ocr_quality_item":
                    index, signals, numbers = payload
                    numbered = ", ".join(str(int(n)) for n in tuple(numbers)[:8])
                    self._log(
                        f"영상 {int(index)}번 · 검토 신호 {int(signals)}건"
                        + (f" · 자막 번호 {numbered}" if numbered else "")
                    )
                    continue
                if kind == "ocr_quality_bound_hints":
                    self._quality_review_hints = payload
                    continue
                if kind == "ocr_quality_bound_unavailable":
                    self._log(
                        "검토 신호 번호는 표시하지만 영상·SRT 일치 근거가 없어 "
                        "'추출 후 검토 권장' 바로 열기는 보류됩니다."
                    )
                    continue
                if kind == "ocr_quality_unavailable":
                    self._log("추출 후 품질 신호 확인 불가 · 기존 자막 파일 보존")
                    continue
                if kind == "batch_ocr_quality":
                    self._last_batch_quality_review = tuple(payload)
                    continue
                if kind == "managed_caption_update_done":
                    self._managed_caption_update_running = False
                    continue
                if kind == "managed_caption_update_status":
                    outcome = str(payload)
                    status = {
                        "updated": "공식 yt-dlp 새 버전 검증 완료 · 다음 자막 요청부터 사용",
                        "current": "공식 yt-dlp 안정판이 이미 최신입니다",
                        "not_due": "오늘 공식 yt-dlp 확인을 이미 수행했습니다",
                        "disabled": "자동 갱신이 비활성화돼 번들 버전을 유지합니다",
                    }.get(outcome, "공식 yt-dlp 확인 보류 · " + outcome.split(":", 1)[0])
                    self._log(status)
                    continue
                if kind == "temporal_review_ready":
                    # Worker owns the exact original source and SRT identity.
                    # No raw OCR transcript or images enter the GUI event.
                    self._temporal_review_hints = payload
                    continue
                if kind == "native_split_review_ready":
                    # Same source/SRT SHA attestation used for temporal review.
                    # Stored only in GUI session RAM; discarded on new run.
                    self._native_split_review_hints = payload
                    continue
                if kind == "local_ocr_preview_ready":
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", maximum=100, value=100)
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    if getattr(self, "ocr_preview_button", None) is not None:
                        self.ocr_preview_button.configure(
                            state="normal" if self.source_mode_var.get() == "local"
                            else "disabled"
                        )
                    self.status_var.set("한 프레임 OCR 확인 완료")
                    self._log("한 프레임 OCR 확인 완료 · 결과는 이 화면에만 표시")
                    try:
                        self._show_local_ocr_preview(payload)
                    except Exception as exc:
                        # Tk/ImageTk may be unavailable even if decoding and
                        # recognition succeeded. Never kill the GUI poll loop
                        # or log private image/text/error details.
                        self.status_var.set("미리보기 화면 표시 실패")
                        self._log("미리보기 화면 표시 실패 · " + type(exc).__name__)
                        messagebox.showerror(
                            "로컬 OCR 미리보기",
                            "화면 생성에 실패했습니다. 원본 영상·자막은 변경되지 않았습니다.",
                        )
                    continue
                if kind == "local_ocr_preview_error":
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", maximum=100, value=0)
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    if getattr(self, "ocr_preview_button", None) is not None:
                        self.ocr_preview_button.configure(
                            state="normal" if self.source_mode_var.get() == "local"
                            else "disabled"
                        )
                    self.status_var.set("한 프레임 OCR 확인 실패")
                    self._log("한 프레임 OCR 확인 실패 · 오류 종류: " + str(payload))
                    messagebox.showerror(
                        "로컬 OCR 미리보기",
                        "선택한 영상 시각·모델 또는 디코더를 확인하세요. "
                        "원본과 이전 자막 결과는 변경되지 않았습니다.",
                    )
                    continue
                if kind == "log":
                    self._log(str(payload))
                elif kind == "tab_capture_ready":
                    self.status_var.set("브라우저 화면 선택 페이지 준비")
                    self._log("로컬 탭 추출 페이지: " + str(payload))
                elif kind == "tab_capture_error":
                    self._tab_capture_outcome_received=True
                    self.status_var.set("브라우저 화면 추출 실패")
                    messagebox.showerror("탭 화면 추출 실패", str(payload))
                elif kind == "tab_capture_result":
                    self._tab_capture_outcome_received=True
                    result=payload if isinstance(payload, dict) else {}
                    result_state=result.get("state")
                    if result_state=="completed":
                        srt=str(result.get("srt_path") or "")
                        ass=str(result.get("ass_path") or "")
                        self.status_var.set("브라우저 자막 추출 완료")
                        self._log(
                            f"브라우저 자막 {int(result.get('cue_count') or 0)}개 완료"
                        )
                        self._log("SRT: "+srt)
                        self._log("ASS: "+ass)
                        messagebox.showinfo(
                            "브라우저 자막 추출 완료",
                            "브라우저 화면 녹화에서 자막을 생성했습니다.\n\n"
                            "SRT: "+srt+"\nASS: "+ass,
                        )
                    elif result_state=="failed":
                        self.status_var.set("브라우저 자막 OCR 실패")
                        self._log(
                            "브라우저 OCR 오류 종류: "
                            +str(result.get("error_type") or "UNKNOWN")
                        )
                    else:
                        self.status_var.set("브라우저 화면 추출 중단 · 새 자막 없음")
                        self._log("브라우저 화면 기록이 완료되지 않아 자막 미생성")
                elif kind == "tab_capture_closed":
                    self.tab_capture_button.configure(state="normal")
                    self.tab_capture_stop_button.configure(state="disabled")
                    if self.worker is None or not self.worker.is_alive():
                        self.start_button.configure(state="normal")
                    if not getattr(self, "_tab_capture_outcome_received", False):
                        self.status_var.set("브라우저 화면 추출 세션 종료")
                elif kind == "progress":
                    if isinstance(payload, ProgressEvent):
                        self._display_progress(payload)
                elif kind == "caption_progress":
                    completed, total = payload
                    self.progress_bar.stop()
                    self.progress_bar.configure(
                        mode="determinate", maximum=max(total, 1),
                        value=completed,
                    )
                    active_mode = getattr(
                        self, "_active_source_mode",
                        self.source_mode_var.get(),
                    )
                    stage_label = (
                        "로컬 영상 처리" if active_mode == "local-batch"
                        else "웹 자막 확인"
                    )
                    self.status_var.set(
                        f"{stage_label} · {completed}/{total}회차"
                    )
                elif kind == "caption_verified":
                    self._verified_existing_count = int(payload)
                elif kind == "captions_done":
                    downloaded, needs_ocr, unavailable, existing_preserved, destination = payload
                    verified_existing = getattr(self, "_verified_existing_count", 0)
                    self._verified_existing_count = 0
                    reviewed, review_flagged, review_unavailable = (
                        getattr(self, "_last_batch_quality_review", (0, 0, 0))
                        if getattr(self, "_active_source_mode", None) == "local-batch"
                        else (0, 0, 0)
                    )
                    self._last_batch_quality_review = (0, 0, 0)
                    self.progress_bar.stop()
                    self.progress_bar.configure(
                        mode="determinate", maximum=100, value=100
                    )
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.status_var.set(
                        "일부 미확보" if needs_ocr or unavailable else
                        "기존 파일 확인 필요" if existing_preserved else
                        "완료 · 검토 권장" if review_flagged else "완료"
                    )
                    active_mode = getattr(
                        self, "_active_source_mode",
                        self.source_mode_var.get(),
                    )
                    if active_mode == "local-batch":
                        self._log(
                            f"로컬 영상 신규 완료 {downloaded}개"
                            f" · 구간 누락이 확인된 회차 {needs_ocr}개"
                            f" · 기존 원본 자막 구조 검증 {verified_existing}개"
                            f" · 미검증 파일 보존 {existing_preserved}개"
                            f" · 처리 실패 {unavailable}개"
                            f" · OCR 검토 권장 {review_flagged}개"
                            f" · 품질 확인 불가 {review_unavailable}개"
                        )
                        messagebox.showinfo(
                            "로컬 영상 일괄 처리 결과",
                            f"신규 처리 완료: {downloaded}개 영상"
                            f"\n일부 구간 미인식(자막 파일 보존): {needs_ocr}개 영상"
                            f"\n기존 원본 자막 구조 확인: {verified_existing}개"
                            f" (영상 원본 바이트 동일성은 미검증)"
                            f"\n미검증 파일 보존: {existing_preserved}개"
                            f" (자동 덮어쓰기 안 함)"
                            f"\n처리 실패: {unavailable}개"
                            f"\nOCR 후 검토 권장: {review_flagged}개 (검사 {reviewed}개)"
                            f"\n품질 확인 불가: {review_unavailable}개"
                            f"\n경고가 없어도 글자 정확도가 증명되지는 않습니다."
                            f"\n\n자막 폴더: {destination}"
                            f"\n\n각 영상의 내장/별도 원본 텍스트 자막을 우선 확인했고,"
                            f" 없을 때만 선택 언어 OCR을 실행했습니다.",
                        )
                    else:
                        self._log(
                            f"자막 신규 확보 {downloaded}개 · 기존 완전성 확인 {verified_existing}개"
                            f" · 검증 불가 파일 보존 {existing_preserved}개 · OCR 필요 {needs_ocr}개"
                            f" · 접근 불가 {unavailable}개"
                        )
                        messagebox.showinfo(
                            "웹 자막 확인 결과",
                            f"신규 확보: {downloaded}개 언어 트랙"
                            f"\n기존 결과 구조 검증: {verified_existing}개"
                            f"\n검증 불가 파일 보존: {existing_preserved}개 (변경하지 않음)"
                            f"\nOCR 필요: {needs_ocr}개"
                            f"\n접근 불가: {unavailable}개"
                            f"\n\n자막 폴더: {destination}"
                            f"\n\n자동 영상 OCR: {'사용' if getattr(self, '_requested_video_ocr_enabled', False) else '사용 안 함'}"
                            f" (실제 자막이 없고 공개 영상에 접근 가능한 경우에만 시도).",
                        )
                elif kind == "cancelled":
                    if getattr(self, "ocr_preview_button", None) is not None:
                        self.ocr_preview_button.configure(
                            state="normal" if self.source_mode_var.get() == "local"
                            else "disabled"
                        )
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", maximum=100, value=0)
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.status_var.set("취소됨")
                    self._log("자막 추출을 안전하게 중단했습니다.")
                elif kind == "partial_done":
                    # Files are kept, but zero-coverage subtitle events are
                    # not confused with a complete episode. UI only displays
                    # counts and the user-selected output paths, never text.
                    srt, ass, missed = payload
                    self.progress_bar.stop()
                    self.progress_bar.configure(
                        mode="determinate", maximum=100, value=100,
                    )
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.status_var.set("일부 미확보")
                    self._log(f"자막 구간 {missed}개 미인식 · 결과 파일 보존")
                    messagebox.showwarning(
                        "일부 자막 미확보",
                        f"자막을 일부 추출했지만 {missed}개 구간이 인식되지 않았습니다."
                        f"\n자막 파일은 보존했습니다.\n\nSRT: {srt}\nASS: {ass}"
                        "\n\n인식된 글자의 정확성은 별도 검증이 필요합니다.",
                    )
                elif kind == "review_done":
                    result, output = payload
                    self.progress_bar.stop()
                    self.progress_bar.configure(
                        mode="determinate", maximum=100, value=100,
                    )
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    if result["status"] == "NO_REVIEW_LOCATIONS":
                        self.status_var.set("검토 경고 없음")
                        messagebox.showinfo(
                            "원본 화면 검토",
                            "현재 범위에는 확인할 자막 경고가 없습니다."
                            " 새로운 파일을 만들지 않았습니다.",
                        )
                    else:
                        self.review_after_var.set(str(result["last_cue"]))
                        self.status_var.set("원본 화면 검토 완료")
                        self._log(
                            f"검토 화면 {result['frame_count']}장"
                            f" · 다음 시작은 자막 {result['last_cue']}번 이후"
                        )
                        messagebox.showinfo(
                            "원본 화면 검토 완료",
                            f"검토한 자막: {result['cue_count']}개\n"
                            f"원본 영상 프레임: {result['frame_count']}장\n\n"
                            f"로컬 결과 폴더: {output}\n\n"
                            "개인정보가 포함될 수 있으니 HTML·이미지 공유에 유의하세요."
                            " 다른 자막을 계속 보려면 검토 버튼을 다시 누르세요.",
                        )
                        try:
                            if _open_owner_review_page(output):
                                self._log("원본 프레임·OCR 문장 오프라인 검토 페이지를 열었습니다.")
                            else:
                                self._log("검토 페이지: 개인 결과 폴더의 review.html")
                        except (OSError, ValueError):
                            self._log("검토 페이지 자동 열기 실패 · 개인 결과 폴더는 보존")
                elif kind == "done":
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", maximum=100, value=100)
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    warning_count = getattr(self, "_last_local_ocr_quality_signals", 0)
                    sample_numbers = getattr(self, "_last_local_quality_samples", ())
                    if (warning_count and getattr(self, "_quality_review_hints", None)
                            and getattr(self, "review_mode_var", None) is not None
                            and self.review_mode_var.get() == "경고 자막만"):
                        self.review_mode_var.set("추출 후 검토 권장")
                    self._last_local_ocr_quality_signals = 0
                    self._last_local_quality_samples = ()
                    self.status_var.set(
                        "완료 · 검토 권장" if warning_count else "완료"
                    )
                    srt, ass = payload
                    guidance = (
                        f"\n\n원본 화면 검토 권장 신호: {warning_count}건"
                        + ("\n참고 자막 번호: " + ", ".join(str(n) for n in sample_numbers)
                           if sample_numbers else "")
                        + "\n'원본 화면 검토 (수동)'에서 확인할 수 있습니다."
                        if warning_count else ""
                    )
                    messagebox.showinfo(
                        "완료",
                        f"자막 생성이 완료되었습니다.\n\nSRT: {srt}\nASS: {ass}"
                        + guidance + "\n\n자동 검사로 인식 글자의 정확성을 증명하지는 않습니다.",
                    )
                elif kind == "error":
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate", maximum=100, value=0)
                    self.cancel_button.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.status_var.set("실패")
                    if isinstance(payload, dict):
                        summary = str(payload.get("summary") or "알 수 없는 오류")
                        detail = str(payload.get("traceback") or summary)
                    else:
                        detail = str(payload)
                        summary = detail.strip().splitlines()[-1] if detail.strip() else "알 수 없는 오류"
                    self._log(detail)
                    messagebox.showerror(
                        "작업 실패",
                        f"{summary}\n\n상세 내용은 진행 로그를 확인하세요.",
                    )
        except queue.Empty:
            pass
        active_tab=getattr(self, "_tab_capture_thread", None)
        if (getattr(self, "_pending_close", False)
                and (self.worker is None or not self.worker.is_alive())
                and (active_tab is None or not active_tab.is_alive())):
            self.progress_bar.stop()
            self.root.destroy()
            return
        self.root.after(100, self._poll_events)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    ExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
