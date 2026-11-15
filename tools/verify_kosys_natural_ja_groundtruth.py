"""REJECTED visual-ground-truth experiment: Kosys Japanese audio TimedText.

The pinned Wikimedia video is genuine authored media, but inspection of the
actual published frames established that its burned-in Japanese display text
does NOT match the separately published Japanese TimedText. That TimedText is
an audio/narration transcript, not ground truth for the visible hard text.
Historical correlation numbers remain reproducible evidence only; they MUST
NOT be interpreted as hard-subtitle OCR accuracy, CER, recall, or timing.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, urllib.request
from pathlib import Path

from compare_external_engine_exports import Cue, evaluate, parse_strict_srt
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

MEDIA_URL=("https://upload.wikimedia.org/wikipedia/commons/3/3a/"
"%E3%81%93%E3%81%86%E3%81%97%E3%81%99%EF%BC%8115%E7%A7%92CM.webm")
MEDIA_BYTES=3529418
MEDIA_SHA1="5ebfe35affd1298f2b1a470a733df1ce98d24d34"
REFERENCE_URL=("https://commons.wikimedia.org/w/index.php?title=TimedText:"
"%E3%81%93%E3%81%86%E3%81%97%E3%81%99%EF%BC%8115%E7%A7%92CM.webm.ja.srt"
"&oldid=1143770027&action=raw")
REFERENCE_REVISION=1143770027
EXPECTED_TIMES=((.120,2.120),(3.660,4.800),(4.920,6.720),(7.240,9.240),
                (9.720,10.470),(11.080,13.040),(13.860,14.820))
EXPECTED_TEXT_SHA256="7083c2201370f518817e59f1bc158517b7854709e239a317ab0bc1e614bf1228"
MAX_MEDIA=4000000
MAX_REF=16384

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise ValueError("unapproved public-source redirect")

def _opener(): return urllib.request.build_opener(NoRedirect())

def _reference():
    req=urllib.request.Request(REFERENCE_URL,headers={
        "User-Agent":"HSE-Natural-JA-Independent-TimedText/1.0",
        "Accept-Encoding":"identity"})
    with _opener().open(req,timeout=30) as response:
        if response.status!=200 or response.geturl()!=REFERENCE_URL:
            raise ValueError("unexpected TimedText response")
        raw=response.read(MAX_REF+1)
    if not 100<len(raw)<=MAX_REF: raise ValueError("reference size changed")
    cues=parse_strict_srt(raw)
    if len(cues)!=7 or any(abs(c.start-a)>.001 or abs(c.end-b)>.001
                           for c,(a,b) in zip(cues,EXPECTED_TIMES)):
        raise ValueError("reference timings changed")
    digest=hashlib.sha256("\x1e".join(c.text for c in cues).encode()).hexdigest()
    if digest!=EXPECTED_TEXT_SHA256: raise ValueError("reference text changed")
    return cues,hashlib.sha256(raw).hexdigest()

def _media(path):
    if path.exists() or path.is_symlink(): raise FileExistsError("media scratch exists")
    stage=path.with_suffix(".partial")
    h1,h256=hashlib.sha1(),hashlib.sha256(); size=0
    req=urllib.request.Request(MEDIA_URL,headers={
        "User-Agent":"HSE-Natural-JA-Independent-TimedText/1.0",
        "Accept-Encoding":"identity"})
    try:
        with _opener().open(req,timeout=45) as response, stage.open("xb") as out:
            if response.status!=200 or response.geturl()!=MEDIA_URL:
                raise ValueError("unexpected media response")
            declared=response.headers.get("Content-Length")
            if declared is not None and (not declared.isdecimal() or int(declared)!=MEDIA_BYTES):
                raise ValueError("media length header changed")
            while True:
                chunk=response.read(65536)
                if not chunk: break
                size+=len(chunk)
                if size>MAX_MEDIA or size>MEDIA_BYTES: raise ValueError("media too large")
                h1.update(chunk);h256.update(chunk);out.write(chunk)
        if size!=MEDIA_BYTES or h1.hexdigest()!=MEDIA_SHA1:
            raise ValueError("official media pin mismatch")
        stage.rename(path); return h256.hexdigest()
    except BaseException:
        stage.unlink(missing_ok=True); raise

def _probe(path):
    p=subprocess.run([resolve_binary("ffprobe"),"-v","error","-protocol_whitelist","file,pipe",
        "-select_streams","v:0","-show_entries","stream=codec_name,width,height:format=duration",
        "-of","json",str(path)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,timeout=30,check=False)
    if p.returncode or len(p.stdout)>32768: raise RuntimeError("probe failed")
    data=json.loads(p.stdout); streams=data.get("streams",[]); duration=float(data["format"]["duration"])
    if (len(streams)!=1 or streams[0].get("codec_name")!="vp9"
        or streams[0].get("width")!=854 or streams[0].get("height")!=478
        or not math.isfinite(duration) or not 15.0<duration<15.2):
        raise ValueError("published media metadata changed")
    return round(duration,3)

def _ocr(video,out,models,mode):
    adaptive=AdaptiveOCREngine(RapidOCRAdapter(language="ja",model_dir=str(models)))
    engine=adaptive if mode=="baseline" else LightCaptionEngine(adaptive)
    report=extract(str(video),engine,out,streaming=True)
    observed=parse_strict_srt(report.srt_path.read_bytes())
    fmt,ass=parse_subtitle_bytes(report.ass_path.read_bytes(),source_url=report.ass_path.name)
    if fmt!="ass" or len(observed)!=len(ass) or len(observed)!=report.cue_count:
        raise RuntimeError("SRT/ASS count mismatch")
    if any(a.text!=b.text or abs(a.start-b.start)>.015 or abs(a.end-b.end)>.015
           for a,b in zip(observed,ass)):
        raise RuntimeError("SRT/ASS content mismatch")
    return observed,{"segments":report.segment_count,"cues":report.cue_count,
                     "ocr_calls":adaptive.total_calls,"retry_calls":adaptive.retry_calls}

def run(root,models):
    if root.exists() or root.is_symlink() or not models.is_dir():
        raise ValueError("fresh scratch and pinned models required")
    root.mkdir(mode=0o700)
    truth,ref_sha=_reference(); video=root/"source.webm"; media_sha=_media(video)
    duration=_probe(video); variants={}
    for mode in ("baseline","light"):
        observed,execution=_ocr(video,root/mode,models,mode)
        score=evaluate(truth,observed,[])
        variants[mode]={**execution,
          "reference_cues":score["reference_cues"],
          "observed_cues":score["extracted_cues"],
          "aligned_cues":score["matched_cues"],
          "missed_reference_cues":score["missed_reference_cues"],
          "extra_observed_cues":score["extra_extracted_cues"],
          "exact_visible_text_ignoring_wrap":score["exact_visible_text_ignoring_wrap"],
          "exact_visible_text_and_wrap":score["exact_visible_text_and_wrap"],
          "strict_both_edges_250ms_and_text_wrap":score["strict_both_edges_250ms_and_text_wrap"],
          "aggregate_character_error_rate":score["aggregate_character_error_rate"],
          "mean_absolute_edge_error_seconds":score["mean_absolute_edge_error_seconds"]}
        if hashlib.sha256(video.read_bytes()).hexdigest()!=media_sha:
            raise RuntimeError("source bytes changed")
    b,l=variants["baseline"],variants["light"]
    return {"schema":1,"source":"OPAP-JP Kosys 15-second commercial",
      "source_exact_size_sha1_verified":True,"source_duration_seconds":duration,
      "source_sha256_unchanged":True,"reference_revision":REFERENCE_REVISION,
      "reference_sha256":ref_sha,"independent_reference_cues":len(truth),
      "publisher_authored_natural_video":True,"synthetic_caption_overlay":False,
      "reference_scope":"audio_narration_timedtext_not_visual_burned_text",
      "hard_subtitle_ground_truth":False,"product_accuracy_metric_valid":False,
      "reference_used_for_inference":False,"reference_independently_frame_verified":False,
      "variants":variants,
      "light_minus_baseline":{
        "aligned_cues":l["aligned_cues"]-b["aligned_cues"],
        "missed_reference_cues":l["missed_reference_cues"]-b["missed_reference_cues"],
        "extra_observed_cues":l["extra_observed_cues"]-b["extra_observed_cues"],
        "exact_visible_text_ignoring_wrap":l["exact_visible_text_ignoring_wrap"]-b["exact_visible_text_ignoring_wrap"],
        "strict_both_edges_250ms_and_text_wrap":l["strict_both_edges_250ms_and_text_wrap"]-b["strict_both_edges_250ms_and_text_wrap"]},
      "production_default_modified":False,"candidate_auto_enabled":False,
      "paid_model_requests":0,"user_media_read":False,
      "release":"REJECTED_REFERENCE_SCOPE"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--approve-cc-source-download",action="store_true");a=p.parse_args()
    if not a.approve_cc_source_download:
        print("HSE_KOSYS_JA=SOURCE_DOWNLOAD_NOT_APPROVED");return 2
    try: result=run(a.scratch,a.model_dir)
    except Exception as exc:
        print("HSE_KOSYS_JA_HOLD "+json.dumps({"error_class":type(exc).__name__,
              "auto_retry":False,"accuracy_proven":False},sort_keys=True));return 3
    print("HSE_KOSYS_JA_RESULT "+json.dumps(result,sort_keys=True));return 0

if __name__=="__main__": raise SystemExit(main())
