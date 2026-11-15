"""Native Matroska PGS/VobSub -> seconv bitmap -> bundled PP-OCRv5.

Isolated original upstream test fixture, exact track language, no installed
program, no user media, no transcription text/paths in CI output. This
qualifies source-to-source integration, not natural-anime OCR accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from hardsub_extractor.local_image_subtitles import acquire_local_embedded_image
from hardsub_extractor.matroska_track_headers import read_image_track_headers
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.runtime_paths import resolve_binary


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(65536), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    a = p.parse_args()
    source = a.fixture.resolve(strict=True)
    pin = sha(source)
    if source.suffix.lower()!=".mkv" or not a.model_dir.is_dir():
        raise ValueError("missing original image source or verified OCR models")
    # The fixture has two image subtitle bitstreams, preserved under remux.
    # English is a test-only tag, never inferred in user source media.
    results=[]
    model = AdaptiveOCREngine(RapidOCRAdapter(
        model_dir=str(a.model_dir),
        manifest_path=str(a.manifest), language="en",
    ))
    with tempfile.TemporaryDirectory(prefix="hse-bdn-ppocr-",dir=a.scratch) as td:
        root = Path(td)
        for n in (0,1):
            remux = root / f"bitmap-{n}.mkv"
            cmd=[resolve_binary("ffmpeg"),"-nostdin","-hide_banner","-loglevel","error",
                 "-protocol_whitelist","file,pipe","-i",str(source),"-map","0:s",
                 "-c:s","copy","-metadata:s:s:0",
                 f"language={'eng' if n==0 else 'jpn'}",
                 "-metadata:s:s:1",f"language={'eng' if n==1 else 'jpn'}",str(remux)]
            p = subprocess.run(cmd,capture_output=True,stdin=subprocess.DEVNULL,timeout=45)
            if p.returncode or not remux.is_file():
                raise AssertionError("independent bitmap-track remux unavailable")
            media_sha = sha(remux)
            available = read_image_track_headers(remux)
            english = [track for track in available if track.language.lower()=="eng"]
            if len(available)!=2 or len(english)!=1:
                raise AssertionError("exact original-language track selection failed")
            base = root / "output" / f"track-{n}"
            produced = acquire_local_embedded_image(
                remux,base,language="en",ocr_factory=lambda:model,
            )
            if produced is None:
                raise AssertionError("PP-OCR bitmap pipeline returned absent")
            if produced.track_number != english[0].track_number:
                raise AssertionError("wrong image subtitle track")
            fmt,srt=parse_subtitle_bytes(produced.srt_path.read_bytes(),source_url="out.srt")
            ass_fmt,ass=parse_subtitle_bytes(produced.ass_path.read_bytes(),source_url="out.ass")
            if fmt!="srt" or ass_fmt!="ass" or not srt or len(srt)!=len(ass):
                raise AssertionError("bitmap images not recognized")
            if any(x.text!=y.text or abs(x.start-y.start)>.02
                   or abs(x.end-y.end)>.02 for x,y in zip(srt,ass)):
                raise AssertionError("PP-OCR text or timing parity failed")
            meta=json.loads(produced.provenance_path.read_text("utf-8"))
            if (meta.get("ocr_engine")!="hse_multilingual_ppocr"
                    or meta.get("original_text_verified") is not False
                    or meta.get("bitmap_timing_precision")!="BDN_frame_quantized"):
                raise AssertionError("PP-OCR provenance conflated with original text")
            coverage = meta.get("bitmap_ocr_event_coverage")
            if (
                not isinstance(coverage, dict)
                or not isinstance(coverage.get("source_bitmap_events"), int)
                or coverage["source_bitmap_events"] < len(srt)
                or coverage.get("recognized_bitmap_events") != len(srt)
                or coverage.get("unrecognized_bitmap_events")
                   != coverage["source_bitmap_events"] - len(srt)
                or coverage.get("all_source_events_recognized")
                   != (coverage["unrecognized_bitmap_events"] == 0)
                or coverage.get("method") != "nonblank_OCR_per_source_bitmap_event_not_text_accuracy"
            ):
                raise AssertionError("real track source bitmap event coverage is invalid")
            originals=[produced.srt_path,produced.ass_path,
                       produced.extracted_track_path,produced.provenance_path]
            result_sha={path:sha(path) for path in originals}
            try:
                acquire_local_embedded_image(remux,base,language="en",ocr_factory=lambda:model)
            except FileExistsError:
                pass
            else:
                raise AssertionError("existing OCR files were overwritten")
            if any(sha(path)!=digest for path,digest in result_sha.items()):
                raise AssertionError("existing OCR outputs changed")
            if sha(remux)!=media_sha or sha(source)!=pin:
                raise AssertionError("original bitmap source media changed")
            results.append({
                "codec": english[0].codec,
                "cues": len(srt),
                "source_events": coverage["source_bitmap_events"],
                "missed_events": coverage["unrecognized_bitmap_events"],
                "full_event_coverage": coverage["all_source_events_recognized"],
            })
    if {v["codec"] for v in results}!={"dvd_subtitle","hdmv_pgs_subtitle"}:
        raise AssertionError("failed to exercise both bitmap codecs")
    print("HSE_BITMAP_PPOCR_REAL="+json.dumps({
        "real_source_image_tracks":sorted([x["codec"] for x in results]),
        "recognized_cues":{x["codec"]:x["cues"] for x in results},
        "source_bitmap_events":{x["codec"]:x["source_events"] for x in results},
        "unrecognized_bitmap_events":{x["codec"]:x["missed_events"] for x in results},
        "all_source_bitmap_events_recognized":{x["codec"]:x["full_event_coverage"] for x in results},
        "hse_english_rapidocr_model_used":True,
        "subtitle_original_text_not_assumed":True,
        "source_untouched_and_existing_outputs_preserved":True,
        "actual_korean_japanese_chinese_model_verified":False,
        "real_anime_quality_verified":False,
    },sort_keys=True,separators=(",",":")))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
