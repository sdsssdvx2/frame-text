"""OCR-free source-identical JA/ZH raised-captions segmentation study.

Reads the existing public synthetic moving MP4 fixture at EXACT tab lower65
ROI. Uses the current detector, optional local signature subbands and bounded
pending/quiet controls only in this isolated research run. No OCR model,
model downloads, user media or change to production defaults.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import hashlib
import json
import tempfile
from unittest.mock import patch

from hardsub_extractor import detector
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.tab_capture import roi_for_region
from compare_motion_subtitle_band import focused_signature
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS

LANGUAGES=("ja","zh")
# Fractions are relative to the pre-cropped lower65% image, not the whole
# screen. Preserve the upper y~.25 and lower y~.42 real glyph bands.
PROFILES={
    "current":(None,None),
    "focus_10_65":((0.10,0.65),None),
    "bounded_625_375":(None,detector.DetectorConfig(
        max_quiet_segment_seconds=0.625,
        max_pending_change_seconds=0.375)),
    "focus_and_bounded":((0.10,0.65),detector.DetectorConfig(
        max_quiet_segment_seconds=0.625,
        max_pending_change_seconds=0.375)),
    # OCR-free opt-in: stable near-neutral bright glyph pixels, no time cap.
    # Motion-only controls must not gain extra OCR observations.
    "persistent_white_3":("persistent_white_3",None),
}



def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for buf in iter(lambda:stream.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()


def observe(source,profile,captioned):
    band,cfg=PROFILES[profile]
    true_signature=detector.subtitle_signature
    if band == "persistent_white_3":
        sig = detector.persistent_white_signature()
    else:
        def sig(image,width=320):
            return focused_signature(
                image,band=band,original_signature=true_signature,width=width)
    with patch.object(detector,"subtitle_signature",new=sig):
        frames=FFmpegPTSFrameSource(str(source),
                                   roi=roi_for_region("lower65"),sample_fps=8.0)
        with closing(iter(frames)) as stream:
            segments=list(detector.iter_segment_frames(stream,cfg))
    if not segments:
        raise RuntimeError("detector_returned_no_visual_segments")
    pts=[frame.pts for segment in segments for frame in segment.samples]
    if not pts:
        raise RuntimeError("detector_collected_no_samples")
    cues=[]
    if captioned:
        for start,end in EXPECTED_SPANS:
            captured=sorted({float(t) for t in pts if start<=t<end})
            cues.append({
                "samples_in_reference_window":len(captured),
                "first_sample_lag_seconds":(
                    round(captured[0]-start,3) if captured else None),
                "last_sample_before_end_seconds":(
                    round(end-captured[-1],3) if captured else None),
                "two_or_more_samples":len(captured)>=2,
                "boundary_coverage_650ms":bool(
                    captured and captured[0]-start<=0.65
                    and end-captured[-1]<=0.65),
            })
    return {
        "profile":profile,"segments":len(segments),
        "sample_observations":len(pts),
        "max_sample_per_segment":max(len(x.samples) for x in segments),
        "reference_windows":cues,
    }


def study():
    result=[]
    for language in LANGUAGES:
        with tempfile.TemporaryDirectory(prefix="hse-raised-seg-"+language+"-") as td:
            root=Path(td)
            pos=root/"captions.mp4"
            neg=root/"no-caption.mp4"
            encode_video(pos,language,style="white_two_line_high")
            encode_video(neg,language,style="white_two_line_high",
                         include_subtitles=False)
            pins=(digest(pos),digest(neg))
            if pins[0]==pins[1]:
                raise RuntimeError("video_control_content_identical")
            profiles={}
            for mode in PROFILES:
                profiles[mode]={
                    "caption":observe(pos,mode,True),
                    "blank":observe(neg,mode,False),
                }
                if pins!=(digest(pos),digest(neg)):
                    raise RuntimeError("source_video_changed")
            base=profiles["current"]
            candidates={}
            for mode in list(PROFILES)[1:]:
                item=profiles[mode]
                coverage=all(
                    x["two_or_more_samples"] and x["boundary_coverage_650ms"]
                    for x in item["caption"]["reference_windows"])
                blank_growth=item["blank"]["sample_observations"]-base["blank"]["sample_observations"]
                candidates[mode]={
                    "both_cues_observed_near_edges":coverage,
                    "extra_blank_observations":blank_growth,
                    "estimated_ocr_sample_delta":(
                        item["caption"]["sample_observations"]-
                        base["caption"]["sample_observations"]),
                    "worth_one_real_model_test":bool(coverage and blank_growth<=0),
                }
            row={"language":language,"caption_clip_same_for_profiles":True,
                 "blank_clip_same_for_profiles":True,
                 "actual_video_pixels_decoded":True,
                 "roi":"lower65","profiles":profiles,"candidate_filters":candidates}
            result.append(row)
            print("HSE_CJK_RAISED_SEG_CASE "+json.dumps(
                row,sort_keys=True),flush=True)
    return {
        "schema":1,"study":"OCR_free_raised_CJK_visual_segmentation",
        "real_user_source":False,"production_changed":False,
        "no_model_called":True,"source_languages":LANGUAGES,
        "profiles_per_language":len(PROFILES),
        "all_profiles_measured":len(result)==len(LANGUAGES),
        "findings":result,
        "not_a_release_gate":True,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.parse_args()
    try:
        result=study()
    except Exception as exc:
        print("HSE_CJK_RAISED_SEG_ERROR "+json.dumps({
            "error_class":type(exc).__name__}),flush=True)
        return 3
    print("HSE_CJK_RAISED_SEG_RESULT "+json.dumps(
        result,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
