from __future__ import annotations

from pathlib import Path
import json

import cv2

from .models import OCRObservation, VisualSegment


class DiagnosticRecorder:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames = self.root / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def record_segment(
        self,
        index: int,
        segment: VisualSegment,
        observations: list[OCRObservation],
        voted: OCRObservation,
    ) -> None:
        samples = []
        for sample_index, frame in enumerate(segment.samples):
            name = f"seg-{index:05d}-sample-{sample_index:02d}-{frame.pts:010.3f}.jpg"
            path = self.frames / name
            cv2.imwrite(
                str(path),
                frame.image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            obs = observations[sample_index] if sample_index < len(observations) else None
            samples.append({
                "pts": frame.pts,
                "file": str(path.relative_to(self.root)).replace("\\", "/"),
                "ocr": {
                    "text": obs.text if obs else "",
                    "confidence": obs.confidence if obs else 0.0,
                    "metadata": obs.metadata if obs and obs.metadata else {},
                },
            })

        self.records.append({
            "index": index,
            "start": segment.start,
            "end": segment.end,
            "duration": max(0.0, segment.end - segment.start),
            "diagnostics": segment.diagnostics or {},
            "samples": samples,
            "voted": {
                "text": voted.text,
                "confidence": voted.confidence,
                "pts": voted.pts,
                "metadata": voted.metadata or {},
            },
        })

    def write(self, *, extra: dict | None = None) -> Path:
        payload = {
            "schema": 2,
            "segments": self.records,
            "extra": extra or {},
        }
        path = self.root / "diagnostics.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
