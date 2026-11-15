"""Opt-in OCR ROI A/B: identical raised two-row JA/ZH animated MP4s.

Three exact tab-region presets, one positive + matched clean video per language.
The *same* PP-OCRv5 detector/recognizer, sampling and streaming SRT/ASS
pipeline run on each region; no reference cues reach source processing.
A successful study proves measurement execution, NOT an improvement nor a
live anime/user-screen quality pass. No customer content, website or paid API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tempfile
import time

from compare_external_engine_exports import Cue as TruthCue, evaluate
from hardsub_extractor.extraction_progress import ExtractionMonitor
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import CAPTION_REGIONS, roi_for_region
from verify_animated_background_ocr_quality import (
    MAX_VIDEO_BYTES, HEIGHT, _multiline_reference, encode_video,
    subtitle_overlay,
)
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from verify_multilang_ocr_synthetic import choose_font

LANGUAGES=("ja","zh")
REGIONS=("bottom40","lower65","lower80")
STYLE="white_two_line_high"
MISSING_CUE="인식된 자막이 없습니다."


def _hash(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _extract_once(video, dest, language, region, models, manifest, *, detector_config=None):
    dest.mkdir()
    engine=AdaptiveOCREngine(RapidOCRAdapter(
        language=language,model_dir=str(models),manifest_path=str(manifest)))
    base=dest/"captions"
    # The baseline must NOT pass an explicit ROI to production. Otherwise an
    # innocent change in the default geometry could make the control false.
    options={"roi":roi_for_region(region)} if region!="bottom40" else {}
    # Only the explicitly supplied research profile changes segment selection.
    # Historical default/ROI benchmark callers remain byte-identical.
    if detector_config is not None:
        options["detector"]=detector_config
    began=time.monotonic()
    try:
        report=extract(str(video),engine,base,
                       sample_fps=8.0,streaming=True,
                       monitor=ExtractionMonitor(exclusive_outputs=True),
                       **options)
    except RuntimeError as error:
        if not str(error).startswith(MISSING_CUE):
            raise
        if base.with_suffix(".srt").exists() or base.with_suffix(".ass").exists():
            raise RuntimeError("empty_cue_published_partial_files")
        return [],{
            "cues":0,"ocr_calls":engine.total_calls,
            "retry_calls":engine.retry_calls,
            "elapsed_seconds":round(time.monotonic()-began,2),
            "srt_ass_equivalent":True,"empty_export_refused":True,
        }
    _,srt=parse_subtitle_bytes(report.srt_path.read_bytes(),source_url="result.srt")
    _,ass=parse_subtitle_bytes(report.ass_path.read_bytes(),source_url="result.ass")
    equivalent=(
        len(srt)==len(ass)==report.cue_count
        and all(a.text==b.text and abs(a.start-b.start)<=0.02
                and abs(a.end-b.end)<=0.02 for a,b in zip(srt,ass))
    )
    if not equivalent:
        raise RuntimeError("srt_ass_export_mismatch")
    return srt,{
        "cues":len(srt),"ocr_calls":engine.total_calls,
        "retry_calls":engine.retry_calls,
        "elapsed_seconds":round(time.monotonic()-began,2),
        "srt_ass_equivalent":equivalent,"empty_export_refused":False,
    }


def _reference(language):
    return [
        TruthCue(start,end,"\n".join(_multiline_reference(text)))
        for (start,end),text in zip(EXPECTED_SPANS,VIDEO_TEXT[language])
    ]


def study(model_dir,manifest):
    all_rows=[]
    for language in LANGUAGES:
        with tempfile.TemporaryDirectory(prefix="hse-roi-ab-"+language+"-") as td:
            root=Path(td)
            caption=root/"raised.mp4"
            clean=root/"same-motion-no-caption.mp4"
            # Prove this actually tests the ROI boundary: some genuine
            # white caption pixels must be ABOVE the default y=0.58 crop,
            # and other caption pixels must remain inside it. Otherwise a
            # successful A/B result would say nothing about the new control.
            glyphs=subtitle_overlay(VIDEO_TEXT[language][0],
                                     choose_font(language),style=STYLE)
            ys=np.flatnonzero(np.any(glyphs[:,:,3]>0,axis=1))
            cut=round(HEIGHT*CAPTION_REGIONS["bottom40"][1])
            if not (len(ys)>0 and ys.min()<cut and ys.max()>cut
                    and ys.min()>round(HEIGHT*CAPTION_REGIONS["lower65"][1])):
                raise RuntimeError("raised_subtitle_does_not_cross_default_crop")
            # Exactly two videos PER language; no per-ROI re-encoding.
            encode_video(caption,language,style=STYLE)
            encode_video(clean,language,style=STYLE,include_subtitles=False)
            if (not 1024<=caption.stat().st_size<=MAX_VIDEO_BYTES
                    or not 1024<=clean.stat().st_size<=MAX_VIDEO_BYTES):
                raise RuntimeError("roi_fixture_video_size_invalid")
            pins=(_hash(caption),_hash(clean))
            if pins[0]==pins[1]:
                raise RuntimeError("caption_and_clean_video_not_distinct")
            for region in REGIONS:
                case=root/region
                case.mkdir()
                positive,p=_extract_once(caption,case/"caption",language,
                                          region,model_dir,manifest)
                negative,n=_extract_once(clean,case/"clean",language,
                                          region,model_dir,manifest)
                # Strictly post-inference: reference text/times cannot
                # influence frame sampling, recognizer or cue boundaries.
                scored=evaluate(_reference(language),positive,negative)
                if pins!=(_hash(caption),_hash(clean)):
                    raise RuntimeError("original_video_modified_during_ab")
                row={
                    "language":language,"profile":region,
                    "region_geometry":CAPTION_REGIONS[region],
                    "same_input_sha_verified":True,
                    "reference_used_only_after_inference":True,
                    "model_name":"pinned_ppocr_v5",
                    "caption":p,"clean":n,"quality":scored,
                }
                all_rows.append(row)
                # Numeric counts/quality only, never the recovered text.
                print("HSE_RAISED_ROI_AB_CASE "+json.dumps(row,sort_keys=True),
                      flush=True)
    if len(all_rows)!=len(LANGUAGES)*len(REGIONS):
        raise RuntimeError("incomplete_roi_trial")
    compared={}
    for language in LANGUAGES:
        rows={r["profile"]:r for r in all_rows if r["language"]==language}
        base=rows["bottom40"]["quality"]
        compared[language]={}
        for region in REGIONS[1:]:
            q=rows[region]["quality"]
            compared[language][region]={
                "new_exact_wrap_cues":q["exact_visible_text_and_wrap"]-
                                       base["exact_visible_text_and_wrap"],
                "new_exact_visible_cues":q["exact_visible_text_ignoring_wrap"]-
                                          base["exact_visible_text_ignoring_wrap"],
                "additional_blank_false_cues":q["blank_video_false_cues"]-
                                              base["blank_video_false_cues"],
                "better_without_more_blank_false_cues":(
                    q["exact_visible_text_and_wrap"]>
                    base["exact_visible_text_and_wrap"]
                    and q["blank_video_false_cues"]<=base["blank_video_false_cues"]
                ),
                "not_auto_promoted":True,
            }
    return {
        "schema":1,"study":"explicit_tab_roi_raised_two_row_cjk",
        "source":"synthetic_moving_MP4_not_real_anime_or_live_site",
        "languages":LANGUAGES,"regions":REGIONS,
        "videos_per_language":2,
        "actual_roi_pipeline":True,
        "sources_same_per_language":True,
        "all_profiles_fully_measured":True,
        "production_defaults_unchanged":True,
        "not_a_release_gate":True,
        "real_user_picker_verified":False,
        "comparison":compared,
        "rows":all_rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,default=Path("models/manifest.json"))
    args=parser.parse_args()
    try:
        result=study(args.model_dir,args.manifest)
    except Exception as error:
        print("HSE_RAISED_ROI_AB_ERROR "+json.dumps({
            "error_class":type(error).__name__}),flush=True)
        return 3
    print("HSE_RAISED_ROI_AB_RESULT "+json.dumps(result,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
