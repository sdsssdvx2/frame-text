from __future__ import annotations

import importlib.util
import re
import statistics
import numpy as np

from .base import OCREngine, normalize_text
from ..models import Frame, OCRObservation
from ..model_assets import ModelAssetError, normalize_ocr_language

_HANGUL_RE = re.compile(r"[가-힣]")
_ENGLISH_RE = re.compile(r"[A-Za-z]")
_HAN_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]")
_CJK_EDGE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uf900-\ufaff]")
_CJK_PUNCTUATION = "、。，。！？：；「」『』（）【】…" 

_SCRIPT_NEUTRAL_RE = re.compile(r"^(?:[A-Za-z0-9]{1,12}[.!?,…]?|[.!?,…])$")


def openvino_runtime_available() -> bool:
    """Cheap packaging check; importing the heavy runtime remains lazy."""
    try:
        return importlib.util.find_spec("openvino") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _script_neutral_fragment(text: str) -> bool:
    """Short numbers, Latin or punctuation can only join anchored dialogue."""
    return 0 < len(text) <= 12 and _SCRIPT_NEUTRAL_RE.fullmatch(text) is not None


def _cjk_edge_glue(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    left, right = previous[-1], current[0]
    left_cjk = _CJK_EDGE_RE.fullmatch(left) is not None
    right_cjk = _CJK_EDGE_RE.fullmatch(right) is not None
    # Retain Latin-to-Latin word boundaries; CJK adjacency has no space.
    return (
        right in _CJK_PUNCTUATION
        or (left_cjk and (right_cjk or (right.isascii() and right.isalnum())))
        or (right_cjk and (left.isascii() and left.isalnum()))
    )


def _script_anchor(text: str, *, require_hangul: bool, language: str | None) -> bool:
    if language == "en":
        return _ENGLISH_RE.search(text) is not None
    if language == "ja":
        return _JAPANESE_RE.search(text) is not None
    if language == "zh":
        return _HAN_RE.search(text) is not None
    return (not require_hangul or _HANGUL_RE.search(text) is not None)

_PUNCT_ONLY_RE = re.compile(r"^[\s.,!?…~:;·ㆍ)\]}>○ㅇ]+$")
# OCR detector boxes move by a few pixels across adjacent frames.  A real
# EpSite duplicated-edge case measured about 22% overlap, while a synthetic
# regression at 14.9% exposed that the old 15% cut-off was too brittle.  Keep
# a small margin without treating ordinary near-touching words as fragments.
_OVERLAP_FRAGMENT_RATIO = 0.12


def _box_geometry(box, image_shape) -> dict[str, float] | None:
    if not image_shape or len(image_shape) < 2:
        return None
    height = float(image_shape[0])
    width = float(image_shape[1])
    if width <= 0 or height <= 0:
        return None
    arr = np.asarray(box, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    return {
        "cx": round((x0 + x1) * 0.5 / width, 6),
        "cy": round((y0 + y1) * 0.5 / height, 6),
        "width": round(bw / width, 6),
        "height": round(bh / height, 6),
        "area": round((bw * bh) / (width * height), 8),
        "aspect": round(bw / bh, 6) if bh > 0 else 0.0,
    }


def _box_record(index: int, box) -> dict | None:
    arr = np.asarray(box, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "index": index,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "cx": (x0 + x1) * 0.5,
        "cy": (y0 + y1) * 0.5,
        "height": y1 - y0,
        "width": x1 - x0,
    }


def _vertical_overlap_ratio(a: dict, b: dict) -> float:
    overlap = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    denom = min(a["height"], b["height"])
    return overlap / denom if denom > 0 else 0.0


def _shared_edge_length(left: str, right: str, max_chars: int = 3) -> int:
    """Find exact text duplicated at the touching edges of overlapping boxes."""
    upper = min(max_chars, len(left), len(right))
    for size in range(upper, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _subtitle_box_lines(
    txts,
    boxes,
    image_shape,
    *,
    require_hangul: bool = True,
    language: str | None = None,
) -> list[list[int]]:
    """Filter OCR detections and group them into visual subtitle lines.

    OCR detectors often split one rendered Korean subtitle line into multiple
    boxes whose top edges differ by a few pixels. Sorting directly by y0 can
    therefore reverse the natural word order. Group boxes by vertical overlap
    first, then sort each line left-to-right.
    """
    if not txts:
        return []

    height = float(image_shape[0]) if image_shape and len(image_shape) >= 2 else 0.0
    width = float(image_shape[1]) if image_shape and len(image_shape) >= 2 else 0.0

    if boxes is None or len(boxes) != len(txts) or width <= 0 or height <= 0:
        indices = [
            i for i, text in enumerate(txts)
            if normalize_text(str(text))
            and _script_anchor(normalize_text(str(text)),
                               require_hangul=require_hangul, language=language)
        ]
        return [[i] for i in indices]

    records: list[dict] = []
    for i, box in enumerate(boxes):
        rec = _box_record(i, box)
        if rec is None:
            continue
        cx = rec["cx"] / width
        cy = rec["cy"] / height
        text = normalize_text(str(txts[i]))
        # Non-Korean recognizers can split a long centered subtitle into
        # tiny edge words ("I", a single kana/Han glyph) and larger central
        # boxes. Retain wider candidates here, then check whole-line center
        # and horizontal continuity below. Korean keeps its established ROI.
        candidate_x = (
            0.04 <= cx <= 0.96
            if language in {"en", "ja", "zh"}
            else 0.18 <= cx <= 0.82
        )
        # Admit upper-row candidates provisionally, but accept them only
        # when a nearby, centered, independently language-anchored lower
        # subtitle row exists. This restores raised two-line dialogue while
        # retaining the historical lower-ROI gate for isolated scene text.
        if not (candidate_x and cy >= 0.10 and text):
            continue
        anchored = _script_anchor(
            text, require_hangul=require_hangul, language=language,
        )
        if anchored:
            rec["script_anchor"] = True
            records.append(rec)
        elif language in {"en", "ja", "zh"} and _script_neutral_fragment(text):
            # Do not accept this as dialogue until it is spatially attached
            # to a separately recognized, language-matching subtitle.
            rec["script_anchor"] = False
            records.append(rec)

    records.sort(key=lambda rec: (rec["cy"], rec["x0"]))
    lines: list[list[dict]] = []
    for rec in records:
        best_line = None
        best_score = 0.0
        for line in lines:
            score = max(_vertical_overlap_ratio(rec, other) for other in line)
            if score >= 0.50 and score > best_score:
                best_line = line
                best_score = score
        if best_line is None:
            lines.append([rec])
        else:
            best_line.append(rec)

    for line in lines:
        line.sort(key=lambda rec: rec["x0"])
    if language in {"en", "ja", "zh"}:
        centered_lines = []
        for line in lines:
            clusters = []
            for rec in line:
                # Split distant credits from the centered subtitle, even if
                # both happen to share a similar vertical baseline.
                if (not clusters or
                        rec["x0"] - clusters[-1][-1]["x1"] > width * 0.12):
                    clusters.append([rec])
                else:
                    clusters[-1].append(rec)
            for cluster in clusters:
                anchors = [rec for rec in cluster if rec["script_anchor"]]
                if not anchors:
                    continue
                # Attach small script-neutral words and punctuation via a
                # spatial chain. A trailing "?" may follow "OK" rather than
                # touching the Japanese anchor itself; direct anchor-only
                # filtering loses that genuine caption punctuation. Require
                # each edge to be on the same line and near a kept box.
                # Nearly coincident boxes are duplicates, not attachments.
                kept = list(anchors)
                pending = [rec for rec in cluster if not rec["script_anchor"]]
                while pending:
                    newly_kept = []
                    for rec in pending:
                        if any(
                            _vertical_overlap_ratio(rec, neighbor) >= 0.5
                            and max(
                                0.0,
                                neighbor["x0"] - rec["x1"],
                                rec["x0"] - neighbor["x1"],
                            ) <= max(width * 0.03, neighbor["height"] * 0.8)
                            and (
                                max(0.0, min(rec["x1"], neighbor["x1"])
                                    - max(rec["x0"], neighbor["x0"]))
                                / min(rec["width"], neighbor["width"])
                            ) < 0.35
                            for neighbor in kept
                        ):
                            newly_kept.append(rec)
                    if not newly_kept:
                        break
                    kept.extend(newly_kept)
                    pending = [rec for rec in pending if rec not in newly_kept]
                kept.sort(key=lambda rec: rec["x0"])
                left = min(rec["x0"] for rec in kept)
                right = max(rec["x1"] for rec in kept)
                center = (left + right) / (2.0 * width)
                if 0.18 <= center <= 0.82:
                    centered_lines.append(kept)
        lines = centered_lines

    # A raised first subtitle row may sit at y=0.16..0.29 *inside the
    # bottom-40% ROI*. The previous global cy>=0.30 filter silently lost
    # it even though the lower row was recognized. Never globally trust
    # upper scene text: require a second, centered and similarly-sized
    # lower line before the upper row can contribute to the transcript.
    def _line_geometry(line):
        xs = [rec["x0"] for rec in line], [rec["x1"] for rec in line]
        left, right = min(xs[0]), max(xs[1])
        return (
            (left + right) / (2.0 * width),
            statistics.median(rec["cy"] for rec in line) / height,
            statistics.median(rec["height"] for rec in line),
        )

    safe_lower = [
        line for line in lines
        if _line_geometry(line)[1] >= 0.30
        and any(rec["script_anchor"] for rec in line)
    ]
    accepted = list(safe_lower)
    for upper in lines:
        ucx, ucy, uh = _line_geometry(upper)
        if ucy >= 0.30 or not (0.10 <= ucy < 0.30):
            continue
        if not (0.18 <= ucx <= 0.82
                and any(rec["script_anchor"] for rec in upper)):
            continue
        for lower in safe_lower:
            lcx, lcy, lh = _line_geometry(lower)
            if (
                0.18 <= lcx <= 0.82
                and abs(lcx - ucx) <= 0.16
                and 0.07 <= lcy - ucy <= 0.34
                and 0.55 <= uh / max(lh, 1e-6) <= 1.8
                # PP-OCRv5 can return accurate *distinct* stacked caption
                # rows whose axis-aligned detection rectangles overlap by
                # 4-7% of this cropped frame. Previously a hard 3.5%-height
                # edge tolerance discarded KO 6/6 and JA 1/9 raw-correct
                # upper rows in real moving synthetic MP4s. All of the
                # centered, script-anchored, spacing and height-ratio gates
                # above still apply. Bound the allowed overhang by BOTH
                # 8% of the crop and 32% of the shorter row's median glyph
                # height; near-duplicate or scene-title boxes must not
                # become dialogue merely because their boundaries touch.
                and min(rec["y0"] for rec in lower)
                >= max(rec["y1"] for rec in upper) - min(
                    height * 0.08, min(uh, lh) * 0.32
                )
            ):
                accepted.append(upper)
                break
    lines = accepted
    lines.sort(key=lambda line: statistics.median(rec["cy"] for rec in line))
    return [[int(rec["index"]) for rec in line] for line in lines]


def _subtitle_box_indices(
    txts,
    boxes,
    image_shape,
    *,
    require_hangul: bool = True,
    language: str | None = None,
) -> list[int]:
    """Backward-compatible flattened reading order for tests/diagnostics."""
    return [
        index
        for line in _subtitle_box_lines(
            txts,
            boxes,
            image_shape,
            require_hangul=require_hangul,
            language=language,
        )
        for index in line
    ]


def _join_visual_line(
    txts, boxes, indices: list[int], *, language: str = "ko",
) -> str:
    parts: list[str] = []
    previous = None
    previous_text = ""
    for i in indices:
        text = normalize_text(str(txts[i]))
        if not text:
            continue
        rec = _box_record(i, boxes[i]) if boxes is not None and i < len(boxes) else None
        if not parts:
            parts.append(text)
        else:
            glue = " "
            punctuation_only = bool(_PUNCT_ONLY_RE.fullmatch(text))
            if punctuation_only:
                glue = ""
            if not punctuation_only and previous is not None and rec is not None:
                overlap = max(0.0, previous["x1"] - rec["x0"])
                denom = min(previous["width"], rec["width"])
                overlap_ratio = overlap / denom if denom > 0 else 0.0
                if overlap_ratio >= _OVERLAP_FRAGMENT_RATIO:
                    # The detector can give neighbouring boxes overlapping
                    # pixels. If both recognitions contain the same edge
                    # syllable(s), those glyphs are the *same rendered text*,
                    # not repeated dialogue: "확인을" + "을해야" should not
                    # become "확인을을해야". Remove the duplicated prefix and
                    # preserve a word boundary for the remaining recognition.
                    shared = _shared_edge_length(previous_text, text)
                    if shared and shared < len(text):
                        text = text[shared:]
                        glue = " "
                    elif shared == len(text):
                        # Entire current recognition is already represented by
                        # the previous overlapping box.
                        continue
                    else:
                        # Pure fragment split with no duplicated glyph, e.g.
                        # "성적" + "인" -> "성적인".
                        glue = ""
            if language in {"zh", "ja"} and _cjk_edge_glue(previous_text, text):
                # CJK script glue is decided AFTER overlapping-box repair:
                # adjacent recognizers sometimes repeat the same Han glyph.
                # Keep Latin word spaces, and preserve genuine repeated Han
                # characters when two separate boxes do not overlap.
                glue = ""
            parts.append(glue + text)
        previous = rec
        previous_text = text
    return normalize_text("".join(parts))


def _recover_english_leading_word(
    engine,
    image,
    boxes,
    indices: list[int],
    line_text: str,
    *,
    text_score: float,
) -> tuple[str, float | None]:
    """Recover a *visible*, missed one-letter English word before a text line.

    The default PP-OCRv5 detector can miss a very narrow standalone "I"/"A"
    even though recognizer-only on the real pixels reads the complete line.
    Run a bounded recognition-only crop for already detected, multi-box
    dialogue lines whose first accepted box sits noticeably right of the
    subtitle window's left side.

    This cannot invent a word from context: the recognition-only output must
    be highly confident and its entire remaining text must exactly match the
    independently detected line. Never apply to Korean, Japanese or Chinese.
    """
    if boxes is None or len(indices) < 2 or not line_text or not isinstance(image, np.ndarray):
        return line_text, None
    height, width = image.shape[:2]
    if height < 35 or width < 150 or len(line_text) < 8:
        return line_text, None
    records = []
    for i in indices:
        if i >= len(boxes):
            return line_text, None
        record = _box_record(i, boxes[i])
        if record is None:
            return line_text, None
        records.append(record)
    left = min(rec["x0"] for rec in records)
    if left / width < 0.17:
        return line_text, None
    right = max(rec["x1"] for rec in records)
    top = min(rec["y0"] for rec in records)
    bottom = max(rec["y1"] for rec in records)
    x0 = max(0, int(left) - max(50, int(width * 0.042)))
    x1 = min(width, int(right) + max(24, int(width * 0.028)))
    y0 = max(0, int(top) - 12)
    y1 = min(height, int(bottom) + 12)
    if x1 - x0 < 50 or y1 - y0 < 24 or x0 >= left - 15:
        return line_text, None
    try:
        output = engine(
            image[y0:y1, x0:x1],
            use_det=False,
            use_cls=False,
            use_rec=True,
            text_score=text_score,
        )
        texts = list(getattr(output, "txts", None) or ())
        scores = list(getattr(output, "scores", None) or ())
    except Exception:
        # The alternate reading is optional. A supported primary OCR line
        # must never be destroyed because an enhancement call failed.
        return line_text, None
    if len(texts) != 1 or len(scores) != 1:
        return line_text, None
    confidence = float(scores[0])
    if not np.isfinite(confidence) or confidence < 0.98:
        return line_text, None
    candidate = normalize_text(str(texts[0]))
    prefix, sep, remainder = candidate.partition(" ")
    if (
        sep
        and len(prefix) == 1
        and prefix.isascii()
        and prefix.isalpha()
        and remainder == line_text
    ):
        return candidate, confidence
    return line_text, None


class RapidOCRAdapter(OCREngine):
    def __init__(
        self,
        *,
        text_score: float = 0.45,
        model_dir: str | None = None,
        manifest_path: str | None = None,
        korean_subtitle_mode: bool | None = None,
        language: str = "ko",
        onnx_intra_threads: int | None = None,
        inference_engine: str = "onnxruntime",
    ):
        # Product "auto" prefers the already-bundled free OpenVINO CPU path
        # but preserves ONNX Runtime as a compatibility fallback. Explicit
        # "openvino" remains fail-closed so callers can demand that backend.
        if inference_engine not in {"auto", "onnxruntime", "openvino"}:
            raise ValueError("unsupported OCR inference engine")
        if inference_engine == "openvino" and not openvino_runtime_available():
            raise RuntimeError(
                "OpenVINO CPU runtime is not bundled; use ONNX Runtime or "
                "install the optional accelerated runtime"
            )
        requested_inference_engine = inference_engine
        selected_inference_engine = (
            "openvino"
            if inference_engine == "auto" and openvino_runtime_available()
            else "onnxruntime" if inference_engine == "auto"
            else inference_engine
        )
        if onnx_intra_threads is not None and (
            type(onnx_intra_threads) is not int
            or not 1 <= onnx_intra_threads <= 16
        ):
            raise ValueError("ONNX intra-op thread opt-in must be 1..16")
        if inference_engine != "onnxruntime" and onnx_intra_threads is not None:
            raise ValueError("ONNX thread override requires ONNX Runtime")
        self.requested_inference_engine = requested_inference_engine
        self.inference_engine = selected_inference_engine
        self.backend_fallback_reason = None
        self.language = normalize_ocr_language(language)
        if self.language != "ko" and korean_subtitle_mode is True:
            raise ValueError("Korean-only text filtering conflicts with foreign OCR language")
        self.korean_subtitle_mode = (
            self.language == "ko"
            if korean_subtitle_mode is None else korean_subtitle_mode
        )
        if self.language != "ko" and not model_dir:
            raise ModelAssetError(
                "foreign-language OCR requires explicitly SHA-pinned local model assets"
            )
        try:
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
        except Exception as exc:
            raise RuntimeError('RapidOCR가 없습니다. pip install -e ".[ocr]" 로 설치하세요.') from exc
        self.text_score = text_score
        engine_type = (
            EngineType.ONNXRUNTIME
            if selected_inference_engine == "onnxruntime"
            else EngineType.OPENVINO
        )
        rec_lang = {
            "ko": LangRec.KOREAN,
            "en": LangRec.EN,
            "ja": LangRec.CH,
            "zh": LangRec.CH,
        }[self.language]
        params = {
            "Det.engine_type": engine_type,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
            "Rec.lang_type": rec_lang,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Global.use_cls": False,
        }
        if onnx_intra_threads is not None:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = (
                onnx_intra_threads
            )

        if model_dir:
            from ..model_assets import resolve_model_paths
            paths = resolve_model_paths(
                model_dir, manifest_path=manifest_path, verify=True,
                language=self.language,
            )
            params.update({
                "Det.model_path": str(paths.detector),
                "Rec.model_path": str(paths.recognizer),
                "Rec.rec_keys_path": str(paths.dictionary),
            })

        try:
            self.engine = RapidOCR(params=params)
        except Exception as exc:
            if not (
                requested_inference_engine == "auto"
                and selected_inference_engine == "openvino"
            ):
                raise
            # Do not alter model paths, language, thresholds, or user files:
            # only retry construction with the established CPU backend.
            params["Det.engine_type"] = EngineType.ONNXRUNTIME
            params["Rec.engine_type"] = EngineType.ONNXRUNTIME
            self.engine = RapidOCR(params=params)
            self.inference_engine = "onnxruntime"
            self.backend_fallback_reason = type(exc).__name__
        # OpenVINO compiles/allocates lazily on some CPUs, so a failure can
        # surface on the first inference rather than at construction. Keep
        # what is needed to switch the *auto* backend to ONNX Runtime once.
        self._auto_onnx_fallback = (
            (RapidOCR, dict(params), EngineType.ONNXRUNTIME)
            if requested_inference_engine == "auto"
            and self.inference_engine == "openvino"
            else None
        )

    def _call_engine(self, image, **kwargs):
        """Run RapidOCR; auto-OpenVINO falls back to ONNX Runtime once."""
        try:
            return self.engine(image, **kwargs)
        except Exception as exc:
            fallback = getattr(self, "_auto_onnx_fallback", None)
            if fallback is None or self.inference_engine != "openvino":
                raise
            rapid_cls, params, onnx_type = fallback
            params["Det.engine_type"] = onnx_type
            params["Rec.engine_type"] = onnx_type
            self.engine = rapid_cls(params=params)
            self.inference_engine = "onnxruntime"
            self.backend_fallback_reason = type(exc).__name__
            self._auto_onnx_fallback = None
            return self.engine(image, **kwargs)

    def recognize_cropped_bitmap(self, frame: Frame) -> OCRObservation:
        """OCR an isolated MKV PGS/VobSub graphic, NOT a video scene.

        The normal recognizer deliberately discards upper/top scene-text
        regions when searching bottom-centered hardcoded video captions.
        On BDN images the background has ALREADY been removed and the
        entire cropped graphic is a subtitle. Reuse the same verified
        detector and language recognizer but accept arbitrary line position.
        This is only a bounded fallback for otherwise blank graphic events.
        """
        image = frame.image
        result = self._call_engine(
            image, use_det=True, use_cls=False, use_rec=True,
            text_score=self.text_score,
        )
        texts = list(result.txts or ())
        scores = list(result.scores or ())
        if not texts:
            return OCRObservation("", 0.0, frame.pts)
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) != len(texts):
            lines = [normalize_text(str(value)) for value in texts]
            lines = [value for value in lines if value]
            conf = [float(x) for x in scores if np.isfinite(x)]
            return OCRObservation(
                "\n".join(lines), sum(conf) / len(conf) if conf else 0.0,
                frame.pts,
            )
        records = []
        for i, box in enumerate(boxes):
            record = _box_record(i, box)
            if record is not None and normalize_text(str(texts[i])):
                records.append(record)
        records.sort(key=lambda item: (item["cy"], item["x0"]))
        groups: list[list[dict]] = []
        for record in records:
            matched = None
            for line in groups:
                if any(
                    _vertical_overlap_ratio(record, previous) >= 0.48
                    or abs(record["cy"] - previous["cy"])
                    <= min(record["height"], previous["height"]) * 0.42
                    for previous in line
                ):
                    matched = line
                    break
            if matched is None:
                groups.append([record])
            else:
                matched.append(record)
        groups.sort(key=lambda line: sum(x["cy"] for x in line) / len(line))
        lines = []
        confs = []
        for group in groups:
            indices = [
                rec["index"] for rec in sorted(group, key=lambda rec: rec["x0"])
            ]
            value = _join_visual_line(
                texts, boxes, indices, language=self.language,
            )
            if value:
                lines.append(value)
            for i in indices:
                if i < len(scores):
                    score = float(scores[i])
                    if np.isfinite(score):
                        confs.append(score)
        return OCRObservation(
            "\n".join(lines),
            sum(confs) / len(confs) if confs else 0.0,
            frame.pts,
            metadata={"bitmap_only": True, "accepted_boxes": len(records)},
        )

    def recognize(self, frame: Frame) -> OCRObservation:
        # BrowserJPEGFrame.image decodes JPEG on each access. Decode once
        # per inference, reuse those pixels for box geometry and OCR, then
        # release the array instead of caching it on the frame.
        image = frame.image
        result = self._call_engine(
            image,
            use_det=True,
            use_cls=False,
            use_rec=True,
            text_score=self.text_score,
        )
        txts = list(result.txts or ())
        scores = list(result.scores or ())
        if not txts:
            return OCRObservation("", 0.0, frame.pts)

        boxes = getattr(result, "boxes", None)
        language = getattr(self, "language", "ko")
        line_groups = _subtitle_box_lines(
            txts,
            boxes,
            image.shape,
            require_hangul=self.korean_subtitle_mode,
            language=language if language != "ko" else None,
        )
        if not line_groups:
            return OCRObservation("", 0.0, frame.pts)

        lines: list[str] = []
        confs: list[float] = []
        detections: list[dict] = []
        leading_glyph_repairs = 0
        for line_number, group in enumerate(line_groups):
            line_text = _join_visual_line(
                txts, boxes, group, language=language,
            )
            if language == "en" and line_text:
                line_text, recovered_score = _recover_english_leading_word(
                    self.engine, image, boxes, group, line_text,
                    text_score=self.text_score,
                )
                if recovered_score is not None:
                    leading_glyph_repairs += 1
                    confs.append(recovered_score)
            if line_text:
                lines.append(line_text)
            for i in group:
                text = normalize_text(str(txts[i]))
                if not text:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                if i < len(scores):
                    confs.append(score)
                if boxes is not None and i < len(boxes):
                    geometry = _box_geometry(boxes[i], image.shape)
                    if geometry is not None:
                        detections.append({
                            "text": text,
                            "score": round(score, 6),
                            "line": line_number,
                            **geometry,
                        })
        if not lines:
            return OCRObservation("", 0.0, frame.pts)
        metadata = {"detections": detections} if detections else {}
        if leading_glyph_repairs:
            metadata["english_leading_glyph_recognition_only"] = leading_glyph_repairs
        return OCRObservation(
            "\n".join(lines),
            sum(confs) / len(confs) if confs else 0.0,
            frame.pts,
            metadata=metadata or None,
        )
