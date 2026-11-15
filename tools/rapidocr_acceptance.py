from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from difflib import SequenceMatcher

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from hardsub_extractor.detector import DetectorConfig
from hardsub_extractor.frame_source import NormalizedROI
from hardsub_extractor.ocr.base import comparison_key
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract

def sim(a,b):
    return SequenceMatcher(a=comparison_key(a),b=comparison_key(b)).ratio()

def _srt_seconds(value: str) -> float:
    hhmmss, millis = value.strip().split(",", 1)
    hh, mm, ss = (int(x) for x in hhmmss.split(":"))
    return hh * 3600 + mm * 60 + ss + int(millis) / 1000.0

def parse_srt(path):
    # Minimal parser for our own generated SRT, retaining timings for acceptance.
    blocks=path.read_text(encoding="utf-8").strip().split("\n\n")
    result=[]
    for block in blocks:
        lines=block.splitlines()
        if len(lines)>=3 and " --> " in lines[1]:
            start,end=lines[1].split(" --> ",1)
            result.append({
                "start": _srt_seconds(start),
                "end": _srt_seconds(end),
                "text": "\n".join(lines[2:]),
            })
    return result

def _print_json_utf8(value) -> None:
    text=json.dumps(value,ensure_ascii=False,indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows hosted runners can expose a legacy cp1252 console even
        # though the evidence file itself is UTF-8.  Reporting must never turn
        # a completed OCR inference into a false CI failure.
        buffer=getattr(sys.stdout,"buffer",None)
        if buffer is None:
            print(text.encode("unicode_escape").decode("ascii"))
        else:
            buffer.write(text.encode("utf-8")+b"\n")
            buffer.flush()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--min-average-similarity",type=float,default=.82)
    ap.add_argument("--min-cue-similarity",type=float,default=.55)
    ap.add_argument("--max-start-error",type=float,default=.30)
    ap.add_argument("--max-end-error",type=float,default=.35)
    ap.add_argument("--assert-quality",action="store_true")
    a=ap.parse_args()

    fixture=ROOT/"artifacts/synthetic-korean-hardsub-stress.mp4"
    truth_path=ROOT/"artifacts/synthetic-stress-truth.json"
    if not fixture.exists():
        subprocess.run([sys.executable,str(ROOT/"tools/make_stress_fixture.py")],check=True)

    truth=json.loads(truth_path.read_text(encoding="utf-8"))
    out=ROOT/"artifacts/rapidocr-acceptance"
    model_dir = ROOT / "models" / "runtime"
    adapter = RapidOCRAdapter(model_dir=str(model_dir) if model_dir.exists() else None)
    report=extract(str(fixture),adapter,out,roi=NormalizedROI(),
                   detector=DetectorConfig(stable_frames=2,max_samples_per_segment=3,sample_spacing=.15))
    got=parse_srt(report.srt_path)
    expected=truth["cues"]
    pairs=[]
    for i,want in enumerate(expected):
        actual=got[i] if i<len(got) else {"start":None,"end":None,"text":""}
        similarity=sim(want["text"],actual["text"])
        start_error=(abs(actual["start"]-want["start"]) if actual["start"] is not None else None)
        end_error=(abs(actual["end"]-want["end"]) if actual["end"] is not None else None)
        pairs.append({
            "expected":want["text"],"actual":actual["text"],"similarity":similarity,
            "expected_start":want["start"],"actual_start":actual["start"],"start_error":start_error,
            "expected_end":want["end"],"actual_end":actual["end"],"end_error":end_error,
        })
    avg=sum(x["similarity"] for x in pairs)/len(pairs)
    max_start=max((x["start_error"] for x in pairs if x["start_error"] is not None), default=None)
    max_end=max((x["end_error"] for x in pairs if x["end_error"] is not None), default=None)
    min_cue=min((x["similarity"] for x in pairs),default=0.0)
    result={
        "cue_count_expected":len(expected),"cue_count_actual":len(got),
        "average_similarity":avg,"minimum_cue_similarity":min_cue,
        "max_start_error":max_start,"max_end_error":max_end,
        "thresholds":{
            "min_average_similarity":a.min_average_similarity,
            "min_cue_similarity":a.min_cue_similarity,
            "max_start_error":a.max_start_error,
            "max_end_error":a.max_end_error,
        },
        "pairs":pairs,"srt":str(report.srt_path),"ass":str(report.ass_path)
    }
    dest=ROOT/"artifacts/rapidocr-acceptance.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    _print_json_utf8(result)
    failed = (
        len(got)!=len(expected)
        or avg<a.min_average_similarity
        or min_cue<a.min_cue_similarity
        or max_start is None or max_start>a.max_start_error
        or max_end is None or max_end>a.max_end_error
    )
    if a.assert_quality and failed:
        raise SystemExit(2)

if __name__=="__main__":main()
