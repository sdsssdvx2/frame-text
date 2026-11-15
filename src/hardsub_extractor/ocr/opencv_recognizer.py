from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

from .base import OCREngine, normalize_text
from .line_splitter import split_subtitle_lines
from ..models import Frame, OCRObservation


class OpenCVRecognizerError(RuntimeError):
    pass


def load_ppocr_dictionary(path: str | Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    chars = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if chars and chars[-1] == "":
        chars.pop()
    if not chars:
        raise OpenCVRecognizerError("empty PP-OCR dictionary")
    # Match RapidOCR/PaddleOCR CTCLabelDecode:
    # blank at 0, dictionary, then explicit space token.
    return ["blank", *chars, " "]


def ctc_decode_logits(
    logits: np.ndarray,
    characters: list[str],
) -> tuple[str, float]:
    arr = np.asarray(logits)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise OpenCVRecognizerError(f"unexpected recognizer output shape: {arr.shape}")

    # Expected PP-OCR output is [N, T, C]. Handle [T, N, C] defensively.
    if arr.shape[0] != 1 and arr.shape[1] == 1:
        arr = np.transpose(arr, (1, 0, 2))
    if arr.shape[0] != 1:
        raise OpenCVRecognizerError(f"batch output not supported here: {arr.shape}")

    # Some runtimes return probabilities; others logits. Softmax only when
    # values do not already look like probabilities.
    scores = arr[0].astype(np.float32)
    row_sums = scores.sum(axis=1)
    if (
        np.any(scores < -1e-5)
        or np.any(scores > 1.00001)
        or float(np.mean(np.abs(row_sums - 1.0))) > 0.05
    ):
        scores -= scores.max(axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)

    ids = scores.argmax(axis=1)
    probs = scores.max(axis=1)

    text: list[str] = []
    confs: list[float] = []
    prev = -1
    for idx, prob in zip(ids.tolist(), probs.tolist()):
        if idx == prev:
            continue
        prev = idx
        if idx == 0:
            continue
        if idx < 0 or idx >= len(characters):
            continue
        text.append(characters[idx])
        confs.append(float(prob))

    out = "".join(text)
    confidence = float(np.mean(confs)) if confs else 0.0
    return normalize_text(out), confidence


def preprocess_ppocr_line(
    image: np.ndarray,
    *,
    height: int = 48,
    base_width: int = 320,
    max_width: int = 2048,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise OpenCVRecognizerError("empty text-line image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise OpenCVRecognizerError(f"expected BGR image, got {image.shape}")

    h, w = image.shape[:2]
    ratio = w / max(1.0, float(h))
    wanted = int(math.ceil(height * ratio))
    # PP-OCR recognizers support dynamic width. Keep widths aligned and bounded.
    canvas_w = max(base_width, int(math.ceil(wanted / 32.0) * 32))
    canvas_w = min(max_width, canvas_w)
    resized_w = min(canvas_w, max(1, wanted))

    resized = cv2.resize(image, (resized_w, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5

    canvas = np.zeros((3, height, canvas_w), dtype=np.float32)
    canvas[:, :, :resized_w] = x
    return canvas[None, ...]


@dataclass(slots=True)
class OpenCVPPRecognizer:
    model_path: str | Path
    dictionary_path: str | Path
    input_height: int = 48
    base_width: int = 320
    max_width: int = 2048

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.dictionary_path = Path(self.dictionary_path)
        if not self.model_path.is_file():
            raise OpenCVRecognizerError(f"recognizer model missing: {self.model_path}")
        if not self.dictionary_path.is_file():
            raise OpenCVRecognizerError(f"dictionary missing: {self.dictionary_path}")
        self.characters = load_ppocr_dictionary(self.dictionary_path)
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except Exception as exc:
            raise OpenCVRecognizerError(
                f"OpenCV could not load recognizer ONNX: {self.model_path}"
            ) from exc

    def recognize_line(self, image: np.ndarray) -> OCRObservation:
        blob = preprocess_ppocr_line(
            image,
            height=self.input_height,
            base_width=self.base_width,
            max_width=self.max_width,
        )
        self.net.setInput(blob)
        out = self.net.forward()
        text, confidence = ctc_decode_logits(out, self.characters)
        return OCRObservation(text, confidence, 0.0)


class OpenCVSubtitleRecognizer(OCREngine):
    """
    Subtitle-specialized recognizer-only fallback.

    It first splits the already-cropped subtitle ROI into up to two line crops,
    then runs PP-OCRv5 Korean recognition on each line. If line splitting fails,
    the full ROI is attempted as one line.
    """

    def __init__(
        self,
        model_path: str | Path,
        dictionary_path: str | Path,
        *,
        min_line_confidence: float = 0.15,
    ):
        self.rec = OpenCVPPRecognizer(model_path, dictionary_path)
        self.min_line_confidence = min_line_confidence

    def recognize(self, frame: Frame) -> OCRObservation:
        lines = split_subtitle_lines(frame.image, max_lines=2)
        images = [x.image for x in lines] or [frame.image]

        texts: list[str] = []
        confs: list[float] = []
        for image in images:
            obs = self.rec.recognize_line(image)
            if obs.text and obs.confidence >= self.min_line_confidence:
                texts.append(obs.text)
                confs.append(obs.confidence)

        if not texts and lines:
            # Heuristic split may be wrong; one full-ROI retry is cheap.
            obs = self.rec.recognize_line(frame.image)
            if obs.text:
                texts = [obs.text]
                confs = [obs.confidence]

        return OCRObservation(
            "\n".join(texts),
            float(np.mean(confs)) if confs else 0.0,
            frame.pts,
        )
