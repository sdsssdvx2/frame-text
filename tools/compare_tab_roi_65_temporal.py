"""Real PP-OCRv5 test of HSE's actual tab light+temporal mode, JA/ZH.

The baseline and opt-in light-temporal mode read exactly the same generated
high-position two-row 8fps MP4 and identically moving no-caption negative.
This reuses offline_ocr, not a fake text layer or model. Independent fixture
references are used ONLY after output. Never changes default or user's files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from compare_external_engine_exports import Cue as ReferenceCue, evaluate
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import offline_ocr
from verify_animated_background_ocr_quality import (
    encode_video, _multiline_reference,
)
from verify_multilang_video_pipeline import VIDEO_TEXT, EXPECTED_SPANS

LANGUAGES=("ja","zh")
MODES=("all","light-temporal")
MISSING="인식된 자막이 없습니다."


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for buf in iter(lambda:file.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()


def expected_cues(lang):
    return [ReferenceCue(start,end,"\n".join(_multiline_reference(text)))
            for (start,end),text in zip(EXPECTED_SPANS,VIDEO_TEXT[lang])]


def run_one(source,output_base,language,mode):
    start=time.monotonic()
    try:
        report=offline_ocr(
            source,output_base,language=language,caption_mode=mode,
            region_mode="lower65")
    except RuntimeError as exc:
        if not str(exc).startswith(MISSING):
            raise
        if (output_base.with_suffix(".srt").exists()
                or output_base.with_suffix(".ass").exists()):
            raise RuntimeError("empty_video_published_partial_pair")
        return [],{"cue_count":0,"empty_pair_not_published":True,
                   "time_seconds":round(time.monotonic()-start,2)}
    _,srt=parse_subtitle_bytes(report.srt_path.read_bytes(),
                               source_url="result.srt")
    _,ass=parse_subtitle_bytes(report.ass_path.read_bytes(),
                               source_url="result.ass")
    if (len(srt)!=len(ass) or len(srt)!=report.cue_count
            or any(a.text!=b.text or abs(a.start-b.start)>.02
                   or abs(a.end-b.end)>.02 for a,b in zip(srt,ass))):
        raise RuntimeError("srt_ass_not_identical")
    return srt,{"cue_count":len(srt),"empty_pair_not_published":False,
                "srt_ass_identical":True,
                "time_seconds":round(time.monotonic()-start,2)}


def run_study(models,manifest):
    rows=[]
    # These patches change ONLY the scratch model lookup paths, never the
    # recognizer class, weights, network, production application or source.
    with patch("hardsub_extractor.runtime_paths.bundled_model_dir",
               return_value=Path(models)),patch(
               "hardsub_extractor.runtime_paths.bundled_manifest_path",
               return_value=Path(manifest)):
        for lang in LANGUAGES:
            with tempfile.TemporaryDirectory(prefix="hse-cjk-style-"+lang+"-") as td:
                root=Path(td)
                pos=root/"caption.mp4"
                neg=root/"blank.mp4"
                encode_video(pos,lang,style="white_two_line_high")
                encode_video(neg,lang,style="white_two_line_high",
                             include_subtitles=False)
                pins=(file_sha(pos),file_sha(neg))
                if pins[0]==pins[1]:
                    raise RuntimeError("caption_control_identical")
                language_rows=[]
                for mode in MODES:
                    base=root/mode
                    base.mkdir()
                    observed,pos_metrics=run_one(pos,base/"positive",lang,mode)
                    control,neg_metrics=run_one(neg,base/"blank",lang,mode)
                    if pins!=(file_sha(pos),file_sha(neg)):
                        raise RuntimeError("source_bytes_changed")
                    numeric=evaluate(expected_cues(lang),observed,control)
                    row={"language":lang,"mode":mode,"region":"lower65",
                         "positive":pos_metrics,"negative":neg_metrics,
                         "quality":numeric,"same_video_bytes":True,
                         "real_model":True,"reference_not_used_by_OCR":True}
                    rows.append(row);language_rows.append(row)
                    print("HSE_CJK_LIGHT_TEMPORAL_CASE "+json.dumps(
                        row,sort_keys=True),flush=True)
    if len(rows)!=4:
        raise RuntimeError("incomplete_CJK_style_comparison")
    decision={}
    for lang in LANGUAGES:
        control=next(r["quality"] for r in rows
                     if r["language"]==lang and r["mode"]=="all")
        candidate=next(r["quality"] for r in rows
                       if r["language"]==lang and r["mode"]=="light-temporal")
        decision[lang]={
            "baseline_exact":control["exact_visible_text_and_wrap"],
            "candidate_exact":candidate["exact_visible_text_and_wrap"],
            "candidate_missing":candidate["missed_reference_cues"],
            "additional_false_cues":candidate["blank_video_false_cues"]-
                                    control["blank_video_false_cues"],
            "baseline_mean_edge_seconds":control["mean_absolute_edge_error_seconds"],
            "candidate_mean_edge_seconds":candidate["mean_absolute_edge_error_seconds"],
            "candidate_strict_250ms":candidate["strict_both_edges_250ms_and_text_wrap"],
            "not_auto_promoted":True,
        }
    return {"schema":1,"scope":"synthetic_raised_two_row_moving_MP4",
            "languages":LANGUAGES,"region":"lower65",
            "modes":MODES,"exact_media_per_language":True,
            "real_user_video":False,"live_tab_picker":False,
            "no_paid_model":True,"default_changed":False,
            "comparison":decision,"cases":rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-dir",required=True,type=Path)
    ap.add_argument("--manifest",type=Path,
                    default=Path("models/manifest.json"))
    args=ap.parse_args()
    try:
        result=run_study(args.model_dir,args.manifest)
    except Exception as exc:
        print("HSE_CJK_LIGHT_TEMPORAL_ERROR "+json.dumps({
            "class":type(exc).__name__}),flush=True)
        return 3
    print("HSE_CJK_LIGHT_TEMPORAL_RESULT "+
          json.dumps(result,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
