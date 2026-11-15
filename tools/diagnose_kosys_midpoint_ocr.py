"""REJECTED visual-ground-truth diagnostic for the pinned Kosys Japanese CM.

The seven TimedText cues are narration/audio subtitles, not the video's visible
burned-in Japanese display text. Reference timestamps therefore do not identify
frames containing those same strings. Historical measurements are retained
only to document the invalid-reference discovery and MUST NOT be used for OCR
accuracy, ROI, timing, model, detector, or product-release decisions.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from contextlib import closing
from pathlib import Path

from compare_external_engine_exports import edit_distance
from hardsub_extractor.frame_source import FFmpegPTSFrameSource, NormalizedROI, probe_video
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.base import normalize_text
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.tab_capture import roi_for_region
from verify_kosys_natural_ja_groundtruth import _media, _probe, _reference

REGIONS=("bottom40","lower65","lower80")
LOCAL_PROTOCOLS=["-protocol_whitelist","file,pipe"]


def _compact(text:str)->str:
    return "".join(normalize_text(text).split())


def _best_line_distance(observed:str, truth:str)->tuple[int,int,bool]:
    """Score each contiguous 1..3 recognized lines after inference only."""
    ref=_compact(truth)
    lines=[_compact(x) for x in normalize_text(observed).splitlines() if _compact(x)]
    if not ref:
        raise ValueError("empty reference")
    candidates=[]
    for start in range(len(lines)):
        for width in range(1,min(3,len(lines)-start)+1):
            candidates.append("".join(lines[start:start+width]))
    if not candidates:
        return len(ref),len(ref),False
    best=min(edit_distance(ref,c) for c in candidates)
    return best,len(ref),best==0


def _nearest_frame(video:Path, second:float, roi:NormalizedROI, info)->Frame:
    seek=float(math.floor(second))
    stream=iter(FFmpegPTSFrameSource(
        str(video),roi=roi,sample_fps=8.0,start_seconds=seek,
        max_duration_seconds=second-seek+.5,extra_input_args=LOCAL_PROTOCOLS,
        preprobed_info=info,decode_timeout_seconds=30.0,
    ))
    best=None;best_gap=float("inf")
    with closing(stream):
        for index,item in enumerate(stream):
            if index>=16: break
            actual=seek+item.pts
            gap=abs(actual-second)
            if gap<best_gap:
                best=Frame(actual,item.image.copy());best_gap=gap
            if actual>second+.125: break
    if best is None or best_gap>.19:
        raise RuntimeError("no bounded frame near independent cue midpoint")
    return best


def run(root:Path,models:Path)->dict:
    if root.exists() or root.is_symlink() or not models.is_dir():
        raise ValueError("fresh scratch and pinned models required")
    root.mkdir(mode=0o700)
    truth,ref_sha=_reference()
    video=root/"source.webm";media_sha=_media(video);_probe(video)
    info=probe_video(str(video),extra_input_args=LOCAL_PROTOCOLS,timeout_seconds=25)
    if info.duration is None or not 15.0<info.duration<15.2:
        raise ValueError("unexpected media duration")

    raw_adapter=RapidOCRAdapter(language="ja",model_dir=str(models))
    region_adapters={
        name:RapidOCRAdapter(language="ja",model_dir=str(models))
        for name in REGIONS
    }
    results=[]
    full=NormalizedROI(0.0,0.0,1.0,1.0)
    for index,cue in enumerate(truth,1):
        midpoint=(cue.start+cue.end)*.5
        full_frame=_nearest_frame(video,midpoint,full,info)
        raw=raw_adapter.recognize_cropped_bitmap(
            Frame(full_frame.pts,full_frame.image.copy())
        )
        raw_edit,ref_chars,raw_exact=_best_line_distance(raw.text,cue.text)
        row={"cue":index,"reference_characters":ref_chars,
             "raw_full_frame_best_edit":raw_edit,
             "raw_full_frame_exact_line":raw_exact}
        for region,adapter in region_adapters.items():
            frame=_nearest_frame(video,midpoint,roi_for_region(region),info)
            observed=adapter.recognize(Frame(frame.pts,frame.image.copy()))
            edit,chars,exact=_best_line_distance(observed.text,cue.text)
            if chars!=ref_chars: raise RuntimeError("reference normalization changed")
            row[region+"_best_edit"]=edit
            row[region+"_exact_line"]=exact
        results.append(row)
        if hashlib.sha256(video.read_bytes()).hexdigest()!=media_sha:
            raise RuntimeError("source bytes changed")
        print("HSE_KOSYS_MIDPOINT_CASE "+json.dumps(row,sort_keys=True),flush=True)

    def summary(prefix):
        edits=[r[prefix+"_best_edit"] for r in results]
        chars=sum(r["reference_characters"] for r in results)
        return {"exact_cues":sum(bool(r[prefix+"_exact_line"]) for r in results),
                "character_edits":sum(edits),
                "reference_characters":chars,
                "best_line_cer":round(sum(edits)/max(1,chars),5)}
    report={"schema":1,
      "diagnostic_only_reference_timestamps_select_frames":True,
      "not_an_extraction_accuracy_score":True,
      "reference_scope":"audio_narration_timedtext_not_visual_burned_text",
      "hard_subtitle_ground_truth":False,"product_accuracy_metric_valid":False,
      "source":"OPAP-JP Kosys 15-second commercial",
      "publisher_authored_natural_video":True,
      "reference_sha256":ref_sha,"reference_cues":len(truth),
      "same_PP_OCRv5_mobile_models":True,"adaptive_retry_used":False,
      "temporal_detector_used":False,"reference_text_used_only_for_post_ocr_scoring":True,
      "results":{
        "raw_full_frame":summary("raw_full_frame"),
        **{region:summary(region) for region in REGIONS}},
      "production_modified":False,"paid_model_requests":0,
      "user_media_read":False,"release":"REJECTED_REFERENCE_SCOPE"}
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--approve-cc-source-download",action="store_true");a=p.parse_args()
    if not a.approve_cc_source_download:
        print("HSE_KOSYS_MIDPOINT=SOURCE_DOWNLOAD_NOT_APPROVED");return 2
    try: out=run(a.scratch,a.model_dir)
    except Exception as exc:
        print("HSE_KOSYS_MIDPOINT_HOLD "+json.dumps({
          "error_class":type(exc).__name__,"auto_retry":False},sort_keys=True));return 3
    print("HSE_KOSYS_MIDPOINT_RESULT "+json.dumps(out,sort_keys=True));return 0

if __name__=="__main__": raise SystemExit(main())
