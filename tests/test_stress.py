import json, subprocess, sys, unittest
from pathlib import Path
from hardsub_extractor.detector import DetectorConfig,segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource,NormalizedROI
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.ocr.base import OCREngine
from hardsub_extractor.cue_builder import merge_cues,recognize_segments

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"


def stress_fixture():
    v=ART/"synthetic-korean-hardsub-stress.mp4"
    j=ART/"synthetic-stress-truth.json"
    if not v.exists() or not j.exists():
        subprocess.run([sys.executable,str(ROOT/"tools/make_stress_fixture.py")],check=True)
    if not v.is_file() or v.stat().st_size == 0:
        raise RuntimeError(f"stress fixture video was not created: {v}")
    if not j.is_file() or j.stat().st_size == 0:
        raise RuntimeError(f"stress fixture truth was not created: {j}")
    return v,json.loads(j.read_text(encoding="utf-8"))


class TruthOCR(OCREngine):
    def __init__(self,cues):self.cues=cues;self.calls=0
    def recognize(self,frame):
        self.calls+=1
        for c in self.cues:
            if c["start"]<=frame.pts<c["end"]:
                return OCRObservation(c["text"],1.0,frame.pts)
        return OCRObservation("",1.0,frame.pts)

class StressAcceptance(unittest.TestCase):
    def test_scene_cuts_short_cue_shadow_two_lines(self):
        v,truth=stress_fixture()
        segments=segment_frames(
            FFmpegPTSFrameSource(str(v),roi=NormalizedROI()),
            DetectorConfig(stable_frames=2,max_samples_per_segment=3,sample_spacing=.15)
        )
        ocr=TruthOCR(truth["cues"])
        cues=merge_cues(recognize_segments(segments,ocr))
        self.assertEqual([c.text for c in cues],[c["text"] for c in truth["cues"]])
        frame=1/truth["fps"]
        for got,want in zip(cues,truth["cues"]):
            self.assertLessEqual(abs(got.start-want["start"]),frame*2.1)
            self.assertLessEqual(abs(got.end-want["end"]),frame*2.1)
        # 240 frames total. OCR stays selective even with scene changes.
        self.assertLess(ocr.calls,80)

if __name__=="__main__":unittest.main()
