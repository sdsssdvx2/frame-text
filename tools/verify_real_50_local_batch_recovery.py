"""Actual FFmpeg 50-MKV original-first batch cancellation and restart study.

Fifty separate local input paths hold copies of two genuine short encoded
MKV fixtures (embedded KO text or video-only + KO SRT sidecar). This does
not claim testing 50 full-length programs, natural subtitle accuracy, or GPU.
Offline, self-contained, isolated; no user data, model, network or native VSF.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hardsub_extractor.extraction_progress import ExtractionCancelled
from hardsub_extractor.local_batch import load_local_batch, run_local_batch
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_local_embedded_video import TEXT, encode_fixture

EPISODES = 50
STOP_AFTER = 17
REF_TIMINGS = ((1.0, 2.5), (3.0, 4.5))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(256 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_inputs(root: Path):
    originals = root / "encoder-source"
    originals.mkdir()
    encoded = encode_fixture(originals)
    no_text = originals / "video-only.mkv"
    process = subprocess.run([
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-protocol_whitelist", "file,pipe",
        "-i", str(encoded), "-map", "0:v:0", "-c:v", "copy",
        str(no_text),
    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL, timeout=90, check=False)
    if process.returncode or no_text.stat().st_size < 1024:
        raise AssertionError("cannot construct real subtitle-free MKV")
    input_folder = root / "inputs"
    input_folder.mkdir()
    videos, tracks = [], []
    for i in range(EPISODES):
        video = input_folder / f"제{i+1:02d}화.1080p.mkv"
        shutil.copyfile(encoded if i % 2 == 0 else no_text, video)
        videos.append(video)
        if i % 2:
            sidecar = video.with_name(video.stem + ".ko.srt")
            sidecar.write_text(
                "1\n00:00:01,000 --> 00:00:02,500\n" + TEXT["ko"][0]
                + "\n\n2\n00:00:03,000 --> 00:00:04,500\n"
                + TEXT["ko"][1] + "\n", encoding="utf-8",
            )
            tracks.append(sidecar)
    return videos, {p: digest(p) for p in (videos + tracks)}


def no_model():
    raise AssertionError("OCR must remain lazy for all 50 original tracks")


def no_native():
    raise AssertionError("do not install/download native helper for soft tracks")


def check_outputs(item, kind):
    base = item.out_base
    srt = base.with_suffix(".srt")
    ass = base.with_suffix(".ass")
    manifest = base.with_name(base.name + ".source.json")
    for file in (srt, ass, manifest):
        if not file.is_file() or file.is_symlink():
            raise AssertionError("real source-first result missing or linked")
    for extension, path in (("srt", srt), ("ass", ass)):
        fmt, cues = parse_subtitle_bytes(
            path.read_bytes(), source_url="test." + extension,
        )
        if fmt != extension or len(cues) != 2:
            raise AssertionError("source-first subtitle count/format invalid")
        for cue, expected, span in zip(cues, TEXT["ko"], REF_TIMINGS):
            if cue.text != expected:
                raise AssertionError("original KO source subtitle text changed")
            if abs(cue.start - span[0]) > 0.08 or abs(cue.end - span[1]) > 0.08:
                raise AssertionError("original KO source subtitle timing changed")
    meta = json.loads(manifest.read_text("utf-8"))
    if (meta.get("mode") != kind or meta.get("cue_count") != 2
        or meta.get("ocr_applied") is not False
        or meta.get("source_video_modified") is not False
        or meta.get("language_requested") != "ko"):
        raise AssertionError("original source metadata invalid")
    if any(str(item.source) in str(v) for v in meta.values()):
        raise AssertionError("source path leaked into provenance")


def snapshot(folder: Path):
    return {
        str(p.relative_to(folder)): digest(p)
        for p in folder.rglob("*") if p.is_file()
    }


def test():
    with tempfile.TemporaryDirectory(prefix="hse-real-50-mkv-") as td:
        root = Path(td)
        sources, original_hashes = prepare_inputs(root)
        output = root / "captions"
        items = load_local_batch(sources, None, output)
        if len(items) != EPISODES or any(x.out_base.suffix for x in items):
            raise AssertionError("50 distinct unambiguous output file names required")
        protected = items[-1].out_base.with_suffix(".srt")
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.write_bytes(b"EXISTING USER SRT NEVER OVERWRITE")
        protected_sha = digest(protected)

        begin = time.perf_counter()
        tracemalloc.start()
        first_results = []
        def interrupt_at_episode(row):
            first_results.append(row)
            if len(first_results) == STOP_AFTER:
                raise ExtractionCancelled("safe synthetic boundary interrupt")
        try:
            run_local_batch(
                items, language="ko", ocr_factory=no_model,
                native_timing_factory=no_native,
                on_result=interrupt_at_episode,
            )
        except ExtractionCancelled:
            pass
        else:
            raise AssertionError("17-episode boundary cancellation was ignored")
        if len(first_results) != STOP_AFTER or any(
            x.mode not in {"local-embedded-text", "local-sidecar-text"}
            for x in first_results
        ):
            raise AssertionError("first 17 real original subtitle outcomes incorrect")
        if any(x.out_base.with_suffix(".srt").exists()
               for x in items[STOP_AFTER:-1]):
            raise AssertionError("cancelled batch continued to write later results")

        resumed = run_local_batch(
            items, language="ko", ocr_factory=no_model,
            native_timing_factory=no_native,
        )
        counts = Counter(x.mode for x in resumed)
        expected = Counter({
            "existing_verified": 17,
            "existing_preserved": 1,
            "local-embedded-text": 16,
            "local-sidecar-text": 16,
        })
        if len(resumed) != EPISODES or counts != expected:
            raise AssertionError("real MKV batch failed to resume from source")
        for i, item in enumerate(items[:-1]):
            check_outputs(
                item, "local-embedded-text"
                if i % 2 == 0 else "local-sidecar-text",
            )
        before_last = snapshot(output)
        final = run_local_batch(
            items, language="ko", ocr_factory=no_model,
            native_timing_factory=no_native,
        )
        if (Counter(x.mode for x in final)
            != Counter({"existing_verified": 49, "existing_preserved": 1})
            or snapshot(output) != before_last):
            raise AssertionError("third pass was not read-only and idempotent")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if peak > 128 * 1024**2:
            raise AssertionError("Python-traced memory exceeded 128MiB")
        if any(digest(p) != previous for p, previous in original_hashes.items()):
            raise AssertionError("original video or source subtitle bytes changed")
        if digest(protected) != protected_sha:
            raise AssertionError("existing user-owned SRT was modified")
        return {
            "source_files": 50,
            "embedded_track_videos": 25,
            "external_sidecar_videos": 25,
            "input_kind": "real encoded short FFmpeg MKV copies",
            "source_text_cues_exactly_matched": 98,
            "cancel_at": STOP_AFTER,
            "resumed_existing_verified": 17,
            "resumed_new_exports": 32,
            "protected_prior_result": 1,
            "third_run_readonly_verified": 49,
            "third_run_preserved": 1,
            "model_and_native_download_calls": 0,
            "source_and_output_SHA_unchanged": True,
            "elapsed_three_passes_seconds": round(time.perf_counter()-begin, 2),
            "peak_python_tracemalloc_mib": round(peak/1024**2, 2),
            "peak_python_only_not_total_process_memory": True,
            "natural_hard_subs_or_50_full_length_videos_proven": False,
            "production_changed": False,
        }


if __name__ == "__main__":
    try:
        result = test()
        print("HSE_REAL_50_BATCH_PASS " + json.dumps(
            result, sort_keys=True, ensure_ascii=True,
        ), flush=True)
    except Exception as err:
        print("HSE_REAL_50_BATCH_FAIL " + json.dumps({
            "error_type": type(err).__name__,
        }), flush=True)
        raise SystemExit(3)
