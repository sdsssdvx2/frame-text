"""REJECTED visual-ground-truth ROI correlation on Kosys audio TimedText.

The reference cues transcribe narration, while the published video contains
different burned-in Japanese display text. Results may describe correlation
with audio timing only and MUST NOT rank HSE hard-subtitle regions or accuracy.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

from compare_external_engine_exports import evaluate
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import roi_for_region
from verify_kosys_natural_ja_groundtruth import _media, _probe, _reference

REGIONS=("bottom40","lower65","lower80")

def _trial(video:Path,out:Path,models:Path,region:str,truth):
    adaptive=AdaptiveOCREngine(RapidOCRAdapter(language="ja",model_dir=str(models)))
    report=extract(str(video),adaptive,out,streaming=True,roi=roi_for_region(region))
    kind,cues=parse_subtitle_bytes(report.srt_path.read_bytes(),source_url=report.srt_path.name)
    kind2,ass=parse_subtitle_bytes(report.ass_path.read_bytes(),source_url=report.ass_path.name)
    if kind!="srt" or kind2!="ass" or len(cues)!=len(ass):
        raise RuntimeError("subtitle export mismatch")
    score=evaluate(truth,cues,[])
    return {
      "region":region,"segments":report.segment_count,"cues":len(cues),
      "ocr_calls":adaptive.total_calls,"retry_calls":adaptive.retry_calls,
      "matched_cues":score["matched_cues"],
      "missed_reference_cues":score["missed_reference_cues"],
      "extra_extracted_cues":score["extra_extracted_cues"],
      "exact_visible_text_ignoring_wrap":score["exact_visible_text_ignoring_wrap"],
      "exact_visible_text_and_wrap":score["exact_visible_text_and_wrap"],
      "strict_both_edges_250ms_and_text_wrap":score["strict_both_edges_250ms_and_text_wrap"],
      "aggregate_character_error_rate":score["aggregate_character_error_rate"],
      "mean_absolute_edge_error_seconds":score["mean_absolute_edge_error_seconds"],
    }

def run(root:Path,models:Path):
    if root.exists() or root.is_symlink() or not models.is_dir():
        raise ValueError("fresh scratch and pinned models required")
    root.mkdir(mode=0o700)
    truth,ref_sha=_reference(); video=root/"source.webm"; media_sha=_media(video)
    _probe(video); rows=[]
    for region in REGIONS:
        row=_trial(video,root/region,models,region,truth);rows.append(row)
        if hashlib.sha256(video.read_bytes()).hexdigest()!=media_sha:
            raise RuntimeError("source bytes changed")
        print("HSE_KOSYS_ROI_CASE "+json.dumps(row,sort_keys=True),flush=True)
    # Result is diagnostic only. Never choose a product region from one CM.
    return {"schema":1,"source":"OPAP-JP Kosys 15-second commercial",
      "publisher_authored_natural_video":True,
      "reference_scope":"audio_narration_timedtext_not_visual_burned_text",
      "hard_subtitle_ground_truth":False,"product_accuracy_metric_valid":False,
      "independent_reference_cues":len(truth),"reference_sha256":ref_sha,
      "reference_used_for_inference":False,"predeclared_regions":list(REGIONS),
      "same_model_and_policy":True,"production_default_modified":False,
      "automatic_region_selection":False,"paid_model_requests":0,
      "user_media_read":False,"results":rows,"release":"REJECTED_REFERENCE_SCOPE"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--approve-cc-source-download",action="store_true");a=p.parse_args()
    if not a.approve_cc_source_download:
        print("HSE_KOSYS_ROI=SOURCE_DOWNLOAD_NOT_APPROVED");return 2
    try: result=run(a.scratch,a.model_dir)
    except Exception as exc:
        print("HSE_KOSYS_ROI_HOLD "+json.dumps({"error_class":type(exc).__name__,
          "auto_retry":False},sort_keys=True));return 3
    print("HSE_KOSYS_ROI_RESULT "+json.dumps(result,sort_keys=True));return 0

if __name__=="__main__": raise SystemExit(main())
