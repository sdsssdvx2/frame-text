from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from hardsub_extractor.visual_sync import (
    estimate_global_offset,
    estimate_piecewise_offsets,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source_video")
    ap.add_argument("target_video")
    ap.add_argument("--max-offset",type=float,default=15.0)
    ap.add_argument("--sample-fps",type=float,default=4.0)
    ap.add_argument("--piecewise",action="store_true")
    ap.add_argument("--window",type=float,default=8.0)
    ap.add_argument("--anchor-spacing",type=float,default=4.0)
    a=ap.parse_args()
    if a.piecewise:
        r=estimate_piecewise_offsets(
            a.source_video,a.target_video,
            max_abs_offset=a.max_offset,sample_fps=a.sample_fps,
            window_seconds=a.window,anchor_spacing=a.anchor_spacing,
        )
        data={
            "mode":"piecewise",
            "anchors":[{
                "source_center":x.source_center,"offset":x.offset,"score":x.score,
                "matched_frames":x.matched_frames,"confidence_margin":x.confidence_margin
            } for x in r.anchors],
            "segments":[{
                "source_start":x.source_start,"source_end":x.source_end,"offset":x.offset,
                "anchor_count":x.anchor_count,"mean_score":x.mean_score
            } for x in r.segments],
            "meaning":"add each segment offset to EpSite subtitle timestamps"
        }
    else:
        r=estimate_global_offset(
            a.source_video,a.target_video,
            max_abs_offset=a.max_offset,sample_fps=a.sample_fps,
        )
        data={
            "mode":"global","offset_seconds":r.offset,"score":r.score,
            "matched_frames":r.matched_frames,"runner_up_score":r.runner_up_score,
            "confidence_margin":r.confidence_margin,
            "meaning":"add offset_seconds to EpSite subtitle timestamps"
        }
    print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
