import json,subprocess,sys,unittest
from pathlib import Path
from hardsub_extractor.detector import DetectorConfig,segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource,NormalizedROI
from hardsub_extractor.models import OCRObservation,Cue
from hardsub_extractor.ocr.base import OCREngine,text_similarity,hangul_jamo_key
from hardsub_extractor.cue_builder import merge_cues,recognize_segments
from hardsub_extractor.writers import write_ass,write_srt

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"
def fixture():
    v=ART/"synthetic-korean-hardsub.mp4";j=ART/"synthetic-truth.json"
    if not v.exists() or not j.exists():subprocess.run([sys.executable,str(ROOT/"tools/make_synthetic_fixture.py")],check=True)
    return v,json.loads(j.read_text(encoding="utf-8"))

class TruthOCR(OCREngine):
    def __init__(self,cues):self.cues=cues;self.calls=0
    def recognize(self,frame):
        self.calls+=1
        for c in self.cues:
            if c["start"]<=frame.pts<c["end"]:return OCRObservation(c["text"],1.0,frame.pts)
        return OCRObservation("",1.0,frame.pts)

class CoreAcceptance(unittest.TestCase):
    def test_synthetic_timing_selective_ocr(self):
        video,truth=fixture()
        segments=segment_frames(FFmpegPTSFrameSource(str(video),roi=NormalizedROI()),
            DetectorConfig(stable_frames=2,max_samples_per_segment=3,sample_spacing=.18))
        ocr=TruthOCR(truth["cues"]);cues=merge_cues(recognize_segments(segments,ocr))
        self.assertEqual([c.text for c in cues],[c["text"] for c in truth["cues"]])
        self.assertEqual(len(cues),3)
        frame=1/truth["fps"]
        for a,e in zip(cues,truth["cues"]):
            self.assertLessEqual(abs(a.start-e["start"]),frame*1.6)
            self.assertLessEqual(abs(a.end-e["end"]),frame*1.6)
        self.assertLess(ocr.calls,40)

    def test_writers(self):
        _,truth=fixture();cues=[Cue(c["start"],c["end"],c["text"]) for c in truth["cues"]]
        s=write_srt(cues,ART/"writer-test.srt").read_text(encoding="utf-8")
        a=write_ass(cues,ART/"writer-test.ass").read_text(encoding="utf-8")
        self.assertIn("00:00:01,000 --> 00:00:02,800",s)
        self.assertIn("두 줄 자막도\n정확히 잡아야 해",s)
        self.assertIn(r"두 줄 자막도\N정확히 잡아야 해",a)

    def test_similarity(self):
        self.assertGreater(text_similarity("괜찮아","괜찮아"),.99)
        self.assertGreater(text_similarity("오늘은 조금 늦었어","오늘은  조금 늦었어"),.99)
        self.assertLess(text_similarity("괜찮아","다른 문장"),.5)

    def test_hangul_jamo_similarity(self):
        # One-vowel OCR error gets a modest boost, but the short-text guard
        # deliberately keeps it well below "same cue" merge territory.
        a=hangul_jamo_key("안돼")
        b=hangul_jamo_key("안되")
        self.assertNotEqual(a,b)
        score=text_similarity("안돼","안되")
        self.assertGreater(score,.50)
        self.assertLess(score,.82)

    def test_short_distinct_dialogue_not_overmerged(self):
        self.assertLess(text_similarity("가자","가라"),.90)
        self.assertLess(text_similarity("맞아","맞지"),.90)

    def test_scene_cut_layout_reorder_is_merged_without_lowering_normal_threshold(self):
        # Real RapidOCR stress-fixture case: a hard scene cut left the same
        # characters on screen but line detection reordered fragments.
        cues=[
            Cue(1.9,2.9,"밝은 장면에서도 보여야 해",0.89),
            Cue(2.9,4.0,"장면에서도 보여야\n밝은\n해",0.96),
        ]
        merged=merge_cues(cues)
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0].text,"밝은 장면에서도 보여야 해")
        self.assertAlmostEqual(merged[0].start,1.9)
        self.assertAlmostEqual(merged[0].end,4.0)

    def test_layout_reorder_guard_does_not_merge_changed_dialogue(self):
        cues=[
            Cue(10.0,11.0,"오늘은 조금 늦었어",0.95),
            Cue(11.0,12.0,"오늘은 아주 늦었어",0.96),
        ]
        self.assertEqual(len(merge_cues(cues)),2)

if __name__=="__main__":unittest.main()
