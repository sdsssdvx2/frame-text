"""Read-only A/B on genuinely published Japanese hard subtitles (Commons).

Creative Commons' original film: CC-BY-2.5; original Japanese subtitles:
Public Domain, credited to Donbraco. This video is NOT constructed by HSE.
There is no human-corrected JA ground-truth timed transcript: comparing cue
counts or OCR text between settings cannot establish accuracy. Download and
outputs remain in disposable CI scratch; no frames, media or words are logged.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile, time
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from hardsub_extractor.frame_source import probe_video
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import offline_ocr

SOURCE_URL = ("https://upload.wikimedia.org/wikipedia/commons/2/28/"
              "Wanna_Work_Together%3F_%28Japanese_subtitled%29.ogv")
SOURCE_PAGE = ("https://commons.wikimedia.org/wiki/"
               "File:Wanna_Work_Together%3F_(Japanese_subtitled).ogv")
PIN_SHA1 = "5d649faac3d7179b796de8f18687ef8f4502de09"
PIN_SIZE = 10055325
START, LENGTH = 35, 65


def obtain_original(path):
    sha = hashlib.sha1()
    count = 0
    with urlopen(Request(SOURCE_URL, headers={
        "User-Agent": "HSE-CC-Japanese-Video-Audit/1.0"
    }), timeout=45) as response, path.open("xb") as writer:
        if response.status != 200 or response.geturl() != SOURCE_URL:
            raise ValueError("official source redirected or changed")
        while True:
            chunk = response.read(min(262144, PIN_SIZE + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > PIN_SIZE:
                raise ValueError("published video exceeds pinned byte limit")
            sha.update(chunk)
            writer.write(chunk)
    if count != PIN_SIZE or sha.hexdigest() != PIN_SHA1:
        raise ValueError("published Commons SHA1 or size mismatch")
    info = probe_video(str(path))
    if info.width != 320 or info.height != 240 or not 179 < (info.duration or 0) < 182:
        raise ValueError("official source duration or dimensions mismatch")
    probe = subprocess.run([resolve_binary("ffprobe"), "-v", "error",
        "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(probe.stdout)["streams"]
    if sum(s.get("codec_type") == "video" for s in streams) != 1 or any(
            s.get("codec_type") == "subtitle" for s in streams):
        raise ValueError("source unexpectedly contains a soft subtitle track")
    return {"original_sha1_verified": True, "original_file_bytes": count,
            "original_dimensions": "320x240",
            "original_duration_seconds": round(info.duration, 3),
            "soft_subtitle_track_present": False}


def make_shared_clip(source, destination):
    command = [resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-ss", str(START), "-i", str(source),
        "-t", str(LENGTH), "-map", "0:v:0", "-an", "-sn",
        "-vf", "scale=640:480:flags=lanczos,setpts=PTS-STARTPTS",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p",
        "-n", str(destination)]
    out = subprocess.run(command, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=120, check=False)
    if out.returncode or not destination.is_file() or not (
            100000 < destination.stat().st_size < 700000000):
        raise RuntimeError("original-derived real video transcode failed")
    info = probe_video(str(destination))
    if (info.width != 640 or info.height != 480 or
            not 62 <= (info.duration or 0) <= 66):
        raise RuntimeError("both-mode source derivative metadata invalid")


def inspect_pair(base):
    srt, ass = base.with_suffix(".srt"), base.with_suffix(".ass")
    if not srt.is_file() or not ass.is_file():
        raise RuntimeError("real original subtitle pair missing")
    _, a = parse_subtitle_bytes(srt.read_bytes(), source_url="result.srt")
    _, b = parse_subtitle_bytes(ass.read_bytes(), source_url="result.ass")
    if not a or len(a) != len(b):
        raise RuntimeError("natural original yielded empty/mismatched cues")
    if any(x.text != y.text or abs(x.start-y.start) > .02
           or abs(x.end-y.end) > .02 for x,y in zip(a,b)):
        raise RuntimeError("published SRT and ASS unequal")
    return a


def measure(model_dir, manifest):
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("SHA-verified local CJK models unavailable")
    with tempfile.TemporaryDirectory(prefix="hse-real-ja-source-") as temp:
        root = Path(temp)
        original, clip = root/"original.ogv", root/"common-clip.mkv"
        provenance = obtain_original(original)
        make_shared_clip(original, clip)
        clip_hash = hashlib.sha256(clip.read_bytes()).hexdigest()
        results = {}
        for mode in ("all", "islands"):
            base = root/("output-"+mode)
            started = time.monotonic()
            with (patch("hardsub_extractor.runtime_paths.bundled_model_dir",
                        return_value=model_dir),
                  patch("hardsub_extractor.runtime_paths.bundled_manifest_path",
                        return_value=manifest)):
                report = offline_ocr(clip, base, language="ja",
                                     region_mode="lower65", caption_mode=mode,
                                     detection_mode="edges")
            lines = inspect_pair(base)
            if report.cue_count != len(lines):
                raise RuntimeError("real original OCR cue count was inconsistent")
            results[mode] = (lines, time.monotonic()-started)
        if hashlib.sha256(clip.read_bytes()).hexdigest() != clip_hash:
            raise RuntimeError("video input changed between comparison modes")
        before, t_before = results["all"]
        after, t_after = results["islands"]
        aligned = lambda x,y: abs(x.start-y.start) <= .8 and abs(x.end-y.end) <= .8
        same = sum(any(aligned(x,y) and x.text == y.text for y in after)
                   for x in before)
        matching_time = sum(any(aligned(x,y) for y in after) for x in before)
        return {**provenance,
            "real_creator_burned_in_Japanese": True,
            "independent_human_reference_available": False,
            "source_segment_start_seconds": START, "source_segment_seconds": LENGTH,
            "both_modes_saw_same_640x480_lossless_intermediate": True,
            "baseline_cue_count": len(before), "islands_cue_count": len(after),
            "baseline_character_count": sum(len(x.text) for x in before),
            "islands_character_count": sum(len(x.text) for x in after),
            "baseline_cues_aligned_in_time_to_islands": matching_time,
            "baseline_cues_exact_same_text_and_nearby_time": same,
            "baseline_seconds": round(t_before, 2),
            "islands_seconds": round(t_after, 2),
            "not_ground_truth_CER_or_superiority": True,
            "paid_api_calls": 0, "operating_files_changed": False,
            "raw_media_frames_and_captions_logged": False, "release": "HOLD"}



def create_known_logo_intervention(clean: Path, marked: Path, folder: Path):
    """Preserve the real authored captions; add only one known static logo.

    The mark is intentionally GENERATED, not claimed to exist in the
    original Commons media. This is a controlled watermark intervention
    on genuinely filmed/published Japanese source pixels.
    """
    from PIL import Image, ImageDraw, ImageFont
    from verify_multilang_ocr_synthetic import choose_font
    import cv2
    import numpy as np
    png=folder/"known-logo.png"
    logo=Image.new("RGBA",(640,480),(0,0,0,0))
    draw=ImageDraw.Draw(logo)
    font=ImageFont.truetype(str(choose_font("ja")),size=34)
    # Bottom 65% OCR region, above the typical lower dialogue lines.
    mark="検証用ロゴ"
    draw.text((188,254),mark,font=font,fill=(255,255,255,230),
              stroke_width=2,stroke_fill=(0,0,0,240))
    logo.save(png,format="PNG")
    if png.stat().st_size < 700:
        raise RuntimeError("rendered known watermark is unexpectedly blank")
    cmd=[resolve_binary("ffmpeg"),"-nostdin","-hide_banner","-loglevel","error",
         "-i",str(clean),"-loop","1","-framerate","8","-i",str(png),
         "-filter_complex",
         "[0:v][1:v]overlay=0:0:shortest=1:format=auto,format=yuv420p[out]",
         "-map","[out]","-an","-t",str(LENGTH),"-c:v","ffv1",
         "-level","3","-pix_fmt","yuv420p","-n",str(marked)]
    result=subprocess.run(cmd,stdin=subprocess.DEVNULL,
                          capture_output=True,timeout=150,check=False)
    if result.returncode or not marked.is_file() or not (
        100000<marked.stat().st_size<700000000):
        raise RuntimeError("bounded real-source watermark intervention unavailable")
    info=probe_video(str(marked))
    if info.width!=640 or info.height!=480 or not 62<= (info.duration or 0) <=66:
        raise RuntimeError("intervention changed real source duration or size")
    a=cv2.VideoCapture(str(clean))
    b=cv2.VideoCapture(str(marked))
    try:
        ok_a,first=a.read()
        ok_b,second=b.read()
    finally:
        a.release()
        b.release()
    if not ok_a or not ok_b:
        raise RuntimeError("watermarked source cannot be decoded")
    delta=cv2.absdiff(first,second)
    # Assert the logo actually appeared in the decoded moving video.
    cropped=delta[248:315,180:470]
    pixels=int(np.count_nonzero(np.max(cropped,axis=2)>18))
    if pixels<220:
        raise RuntimeError("known watermark pixels absent in actual video")
    return {"intervention_only_generated_logo":True,
            "logo_pixels_independently_decoded":pixels,
            "logo_position":"lower65_above_original_caption_band",
            "source_logo_added_in_separate_scratch_derivative":True,
            "pixel_identity_outside_logo_not_independently_certified":True}


def _bottom_two_research(video: Path, output: Path, model_dir: Path, manifest: Path):
    """Reuse the existing experimental HSE wrapper, NOT a second OCR engine.

    The same 8-fps source timeline and lower65 ROI as the other two trial
    arms. No user model installation, frame uploads or text export to logs.
    True 3+ line captions can be lost; no automatic release approval.
    """
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
    from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
    from hardsub_extractor.ocr.line_focus import BottomTwoLineEngine
    from hardsub_extractor.pipeline import extract
    from hardsub_extractor.tab_capture import roi_for_region
    from hardsub_extractor.extraction_progress import ExtractionMonitor

    engine = BottomTwoLineEngine(AdaptiveOCREngine(RapidOCRAdapter(
        language="ja", model_dir=str(model_dir), manifest_path=str(manifest),
    )))
    return extract(
        str(video), engine, output, sample_fps=8.0,
        roi=roi_for_region("lower65"), streaming=True,
        monitor=ExtractionMonitor(exclusive_outputs=True),
    )


def measure_known_logo(model_dir,manifest, *, candidate="islands"):
    """Same author's Japanese video: clean / marked / marked+optional OCR.

    A clean OCR result is a NOISY REFERENCE, not human-transcribed truth.
    The goal is measured contamination and preservation when a known
    competing text layer is added to the real native source.
    """
    if candidate not in {"islands", "bottom-two", "light-temporal"}:
        raise ValueError("unsupported explicit Japanese watermark comparator")
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("SHA-verified local CJK models unavailable")
    with tempfile.TemporaryDirectory(prefix="hse-real-JA-mark-intervention-") as tmp:
        folder=Path(tmp)
        original=folder/"original.ogv"
        clean=folder/"real-clean.mkv"
        marked=folder/"real-with-known-logo.mkv"
        provenance=obtain_original(original)
        make_shared_clip(original,clean)
        logo=create_known_logo_intervention(clean,marked,folder)
        pinned={p:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (original,clean,marked)}
        observations={}
        selected = {
            "bottom-two": "marked_bottom_two",
            "islands": "marked_islands",
            "light-temporal": "marked_light_temporal",
        }[candidate]
        for name,video,mode in [
            ("clean_baseline",clean,"all"),
            ("marked_baseline",marked,"all"),
            (selected,marked,candidate),
            *([("clean_light_temporal",clean,"light-temporal")]
              if candidate == "light-temporal" else []),
        ]:
            start=time.monotonic()
            with (patch("hardsub_extractor.runtime_paths.bundled_model_dir",
                        return_value=model_dir),
                  patch("hardsub_extractor.runtime_paths.bundled_manifest_path",
                        return_value=manifest)):
                if mode == "bottom-two":
                    report = _bottom_two_research(
                        video,folder/("export-"+name),model_dir,manifest
                    )
                else:
                    report=offline_ocr(video,folder/("export-"+name),language="ja",
                                       region_mode="lower65",caption_mode=mode,
                                       detection_mode="edges")
            output=inspect_pair(folder/("export-"+name))
            if report.cue_count!=len(output):
                raise RuntimeError("real video measured cue count differs")
            observations[name]=(output,time.monotonic()-start)
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=v
               for p,v in pinned.items()):
            raise RuntimeError("real source, clean or marked derivative modified")
        clean_cues,t0=observations["clean_baseline"]
        marked_cues,t1=observations["marked_baseline"]
        candidate_cues,t2=observations[selected]
        def near(a,b):
            return abs(a.start-b.start)<=.8 and abs(a.end-b.end)<=.8
        def matching(a,b):
            return sum(any(near(x,y) and x.text==y.text for y in b) for x in a)
        def aligned(a,b):
            return sum(any(near(x,y) for y in b) for x in a)
        marker="検証用ロゴ"
        return {**provenance,**logo,
            "source_clip_seconds":LENGTH,
            "real_authored_original_subtitle":True,
            "known_extra_logo_is_synthetic":True,
            "ground_truth_Japanese_transcript":False,
            "clean_baseline_cues":len(clean_cues),
            "marked_baseline_cues":len(marked_cues),
            selected+"_cues":len(candidate_cues),
            "clean_cues_exact_in_marked_baseline":matching(clean_cues,marked_cues),
            "clean_cues_exact_in_"+selected:matching(clean_cues,candidate_cues),
            "clean_cues_with_nearby_marked_baseline":aligned(clean_cues,marked_cues),
            "clean_cues_with_nearby_"+selected:aligned(clean_cues,candidate_cues),
            "marked_baseline_exact_in_"+candidate.replace("-", "_"):matching(marked_cues,candidate_cues),
            "marked_baseline_logo_literal_cues":sum(marker in x.text for x in marked_cues),
            selected+"_logo_literal_cues":sum(marker in x.text for x in candidate_cues),
            "clean_seconds":round(t0,2),
            "marked_baseline_seconds":round(t1,2),
            selected+"_seconds":round(t2,2),
            "candidate_method":candidate,
            "candidate_three_line_dialogue_may_be_deleted":candidate=="bottom-two",
            "candidate_light_style_rejects_dark_or_colored_subtitles":candidate=="light-temporal",
            "clean_candidate_exact_in_clean_baseline":(
                matching(observations["clean_light_temporal"][0],clean_cues)
                if candidate=="light-temporal" else None),
            "clean_baseline_exact_in_clean_candidate":(
                matching(clean_cues,observations["clean_light_temporal"][0])
                if candidate=="light-temporal" else None),
            "clean_candidate_cues":(
                len(observations["clean_light_temporal"][0])
                if candidate=="light-temporal" else None),
            "clean_candidate_elapsed_seconds":(
                round(observations["clean_light_temporal"][1],2)
                if candidate=="light-temporal" else None),
            "performance_CER_not_provable_without_human_truth":True,
            "OCR_text_or_frames_logged":False,"paid_API_calls":0,
            "operating_files_changed":False,"release":"HOLD"}

def measure_pixel_only(model_dir, manifest):
    """Two isolated real Japanese OCR paths; original text is not ground truth.

    A/B shares the exact same SHA-pinned author-captioned video pixels.
    Pixel-only MUST preserve every baseline cue's text and relative order;
    source timing accuracy cannot be inferred without a human reference.
    """
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("SHA-verified local CJK models unavailable")
    with tempfile.TemporaryDirectory(prefix="hse-real-ja-pixel-only-") as tmp:
        root = Path(tmp)
        original, clip = root / "original.ogv", root / "shared-clip.mkv"
        provenance = obtain_original(original)
        make_shared_clip(original, clip)
        original_sha = hashlib.sha256(clip.read_bytes()).hexdigest()
        rows = {}
        for mode in ("all", "pixel-timing"):
            base = root / ("output-" + mode)
            start = time.monotonic()
            with (
                patch("hardsub_extractor.runtime_paths.bundled_model_dir",
                      return_value=model_dir),
                patch("hardsub_extractor.runtime_paths.bundled_manifest_path",
                      return_value=manifest),
            ):
                report = offline_ocr(clip, base, language="ja", region_mode="lower65",
                                     caption_mode=mode, detection_mode="edges")
            cues = inspect_pair(base)
            if report.cue_count != len(cues):
                raise RuntimeError("real Japanese OCR cue count inconsistent")
            rows[mode] = (cues, time.monotonic()-start)
        if hashlib.sha256(clip.read_bytes()).hexdigest() != original_sha:
            raise RuntimeError("published source-derived pixels modified")
        baseline, baseline_time = rows["all"]
        pixel, pixel_time = rows["pixel-timing"]
        if len(baseline) != len(pixel) or any(
            a.text != b.text or a.confidence != b.confidence
            for a, b in zip(baseline, pixel)
        ):
            raise RuntimeError("pixel-only unexpectedly changed OCR text or cue count")
        deltas = [
            (abs(a.start-b.start), abs(a.end-b.end))
            for a, b in zip(baseline, pixel)
        ]
        return {**provenance,
            "original_author_burned_in_Japanese": True,
            "same_input_pixels_and_PP_OCRv5": True,
            "source_segment_start_seconds": START,
            "source_segment_seconds": LENGTH,
            "baseline_cues": len(baseline),
            "pixel_only_cues": len(pixel),
            "all_text_confidence_and_order_preserved": True,
            "timing_changed_cues": sum(max(x,y) > .001 for x,y in deltas),
            "max_boundary_shift_seconds": round(
                max((max(x,y) for x,y in deltas), default=0.0), 3),
            "baseline_elapsed_seconds": round(baseline_time, 2),
            "pixel_only_elapsed_seconds": round(pixel_time, 2),
            "independent_human_Japanese_transcript": False,
            "timing_accuracy_improvement_not_proven": True,
            "srt_ass_pair_equivalent": True,
            "raw_caption_text_or_frames_logged": False,
            "paid_api_calls": 0, "operating_files_changed": False,
            "release": "HOLD"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--known-logo-intervention", action="store_true")
    parser.add_argument("--known-logo-bottom-two", action="store_true",
                        help="Compare the already existing high-risk last-two-lines opt-in on the same real Japanese watermark fixture")
    parser.add_argument("--pixel-timing-only", action="store_true",
                        help="A/B source-authored Japanese OCR text preservation against optical-only timing")
    parser.add_argument("--known-logo-light-temporal", action="store_true",
                        help="Compare existing bright-text and pixel-timing opt-in on a real Japanese source and its separately marked derivative")
    parser.add_argument("--accept-public-download", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if sum(bool(x) for x in (
        args.known_logo_bottom_two, args.known_logo_intervention,
        args.known_logo_light_temporal, args.pixel_timing_only,
    )) > 1:
        parser.error("select only one optional watermark candidate")
    if args.self_test:
        assert len(PIN_SHA1) == 40 and PIN_SIZE == 10055325 and LENGTH <= 75
        assert "検証用ロゴ" and START >= 0
        print("HSE_REAL_JA_SOURCE_CONTRACT=PASS;NETWORK=OFF;OCR=OFF")
        return 0
    if not args.accept_public_download or args.model_dir is None or args.manifest is None:
        print("HSE_REAL_JA_SOURCE=NO_DOWNLOAD_WITHOUT_EXPLICIT_CONSENT")
        return 2
    try:
        report = measure_pixel_only(args.model_dir, args.manifest) if args.pixel_timing_only else (
            measure_known_logo(
                args.model_dir, args.manifest,
                candidate=("light-temporal" if args.known_logo_light_temporal else
                           "bottom-two" if args.known_logo_bottom_two else "islands"),
            )
            if (args.known_logo_intervention or args.known_logo_bottom_two
                or args.known_logo_light_temporal) else
            measure(args.model_dir,args.manifest)
        )
    except Exception as exc:
        print("HSE_REAL_JA_SOURCE_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "user_media_touched": False,
            "raw_text_or_paths_logged": False, "release": "HOLD"
        }, sort_keys=True))
        return 3
    prefix = (
        "HSE_REAL_JA_PIXEL_ONLY_AB " if args.pixel_timing_only else
        "HSE_REAL_JA_LIGHT_TEMPORAL_AB " if args.known_logo_light_temporal else
        "HSE_REAL_JA_BOTTOM_TWO_AB " if args.known_logo_bottom_two else
        "HSE_REAL_JA_LOGO_AB " if args.known_logo_intervention else
        "HSE_REAL_JA_SOURCE_AB "
    )
    print(prefix + json.dumps(report, sort_keys=True))
    print("HSE_REAL_JA_SOURCE=MEASURED;REAL_ORIGINAL=YES;"
          "HUMAN_TRUTH_CER=UNKNOWN;RELEASE=HOLD;PAID_API=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
