"""Research-only: temporal watermark hypothesis on real Japanese hard subtitles.

Use already-pinned Wikimedia Commons original and add a known synthetic logo
to a SEPARATE lossless copy. Clean source is used only for scoring, never to
select candidates. Frozen HSE PP-OCRv5, detector, streaming/SRT/ASS stay intact.
No original/filtered subtitles, text or frames leave disposable CI scratch.
A noisy clean OCR baseline is NOT independently transcribed ground truth.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile,time
from pathlib import Path
from hardsub_extractor.extraction_progress import ExtractionMonitor
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.tab_capture import roi_for_region
from verify_multiline_streaming_ocr_quality import InMemorySyntheticSegmentAudit
from study_temporal_overlay_quarantine import (
    LineTrace,collect_lines,find_candidates,fold_whitespace,quarantine,
    anonymous_support_summary)
from verify_commons_natural_japanese import (
    START,LENGTH,obtain_original,make_shared_clip,create_known_logo_intervention,
    inspect_pair)


def _measure_clip(video,base,model_dir,manifest):
    audit=InMemorySyntheticSegmentAudit()
    engine=AdaptiveOCREngine(RapidOCRAdapter(
        language="ja",model_dir=str(model_dir),manifest_path=str(manifest)))
    start=time.monotonic()
    report=extract(
        str(video),engine,base,roi=roi_for_region("lower65"),
        sample_fps=8.0,streaming=True,
        diagnostics=audit,monitor=ExtractionMonitor(exclusive_outputs=True))
    cues=inspect_pair(base)
    if report.cue_count!=len(cues):
        raise RuntimeError("genuine Japanese video cue count differs from both exports")
    return cues,audit,round(time.monotonic()-start,2)


def _summary(cues,reference):
    # The clean OCR output is a noisy *comparison*, not verified truth.
    def near(a,b):
        return abs(a.start-b.start)<=0.8 and abs(a.end-b.end)<=0.8
    return {
        "cue_count":len(cues),
        "literal_added_logo_cues":sum("検証用ロゴ" in x.text for x in cues),
        "clean_ocr_exact_text_nearby":sum(
            any(near(x,y) and x.text==y.text for y in cues) for x in reference),
        "clean_ocr_nearby_time":sum(
            any(near(x,y) for y in cues) for x in reference),
        "total_characters":sum(len(x.text) for x in cues),
    }


def compare(model_dir,manifest):
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("missing verified free CJK model assets")
    with tempfile.TemporaryDirectory(prefix="hse-native-ja-temporal-") as temp:
        root=Path(temp)
        original,clean,marked=(root/x for x in
                              ("original.ogv","clean.mkv","with-added-logo.mkv"))
        info=obtain_original(original)
        make_shared_clip(original,clean)
        watermark=create_known_logo_intervention(clean,marked,root)
        pins={p:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (original,clean,marked)}
        clean_cues,clean_audit,clean_s=_measure_clip(
            clean,root/"clean-subtitles",model_dir,manifest)
        marked_cues,marked_audit,marked_s=_measure_clip(
            marked,root/"marked-subtitles",model_dir,manifest)
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=pin
               for p,pin in pins.items()):
            raise RuntimeError("genuine source or research inputs changed")
        marked_observations=collect_lines(marked_audit)
        clean_observations=collect_lines(clean_audit)
        policies={}
        for label,policy in (
            ("folded_three_windows","three_windows"),
            ("folded_strict","strict"),
        ):
            folded=[LineTrace(x.pts,fold_whitespace(x.text),x.cx,x.cy)
                    for x in marked_observations]
            names=find_candidates(folded,LENGTH,flank_policy=policy)
            proposed,changes=quarantine(
                marked_cues,names,normalizer=fold_whitespace)
            policies[label]={
                "candidate_line_count":len(names),
                "candidate_contains_known_mark":any(
                    fold_whitespace("検証用ロゴ") in x for x in names),
                "removed_ocr_lines":changes["removed_lines"],
                "dropped_full_cues":changes["dropped_all_candidate_cues"],
                "hypothetical_only":True,
                "would_produce":_summary(proposed,clean_cues),
            }
        # On clean video, no source-side reference or synthetic mark entered
        # the detector. A proposed deletion here is a warning, not success.
        clean_traces=[LineTrace(x.pts,fold_whitespace(x.text),x.cx,x.cy)
                      for x in clean_observations]
        clean_candidates=find_candidates(
            clean_traces,LENGTH,flank_policy="three_windows")
        return {
            **info,**watermark,
            "original_source":"genuine_author_JA_burned_subtitle",
            "known_watermark":"separately_drawn_synthetic_logo",
            "ground_truth_Japanese_transcript":False,
            "clip_seconds":LENGTH,"clip_begin_seconds":START,
            "same_verified_native_source":True,
            "OCR_extractions":2,"original_source_preserved":True,
            "clean_ocr_seconds":clean_s,"marked_ocr_seconds":marked_s,
            "clean_reference_cues":len(clean_cues),
            "marked_baseline":_summary(marked_cues,clean_cues),
            "clean_three_windows_candidate_count":len(clean_candidates),
            "marked_audited_OCR_lines":len(marked_observations),
            "marked_anonymous_track_support":anonymous_support_summary(
                [LineTrace(x.pts,fold_whitespace(x.text),x.cx,x.cy)
                 for x in marked_observations],LENGTH),
            "policies":policies,"result_is_not_CER_or_proof_of_accuracy":True,
            "subtitle_files_only_original_extracts":True,
            "no_postfiltered_SRT_written":True,"model_output_text_logged":False,
            "paid_API_calls":0,"user_media_touched":False,
            "operating_program_modified":False,"release":"HOLD",
        }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--accept-public-download",action="store_true")
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--manifest",type=Path)
    args=p.parse_args()
    if args.self_test:
        from compare_external_engine_exports import Cue
        probe=[Cue(0,1,"字幕\n検証用ロゴ")]
        a=quarantine(probe,{fold_whitespace("検証用ロゴ")},
                     normalizer=fold_whitespace)
        assert [x.text for x in a[0]]==["字幕"]
        assert probe[0].text=="字幕\n検証用ロゴ"
        assert a[1]["dropped_all_candidate_cues"]==0
        print("HSE_NATURAL_JA_TEMPORAL_CONTRACT=PASS;NETWORK=OFF;OCR=OFF")
        return 0
    if (not args.accept_public_download or args.model_dir is None or
        args.manifest is None):
        print("HSE_NATURAL_JA_TEMPORAL=NO_APPROVAL_NO_NETWORK")
        return 2
    try:
        report=compare(args.model_dir,args.manifest)
    except Exception as exc:
        print("HSE_NATURAL_JA_TEMPORAL_ERROR "+json.dumps({
            "error_class":type(exc).__name__,
            "video_text_or_original_paths_logged":False,
            "operating_app_modified":False,"release":"HOLD",
        },sort_keys=True),flush=True)
        return 3
    print("HSE_NATURAL_JA_TEMPORAL_CASE "+json.dumps(
        report,sort_keys=True),flush=True)
    print("HSE_NATURAL_JA_TEMPORAL=MEASURED;REFERENCE_CER=UNAVAILABLE;"
          "RELEASE=HOLD;USER_MEDIA=NO;PAID_API=NO",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
