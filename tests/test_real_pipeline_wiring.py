"""Real HSE decoding, detection, vote, merging and protected export; deterministic OCR."""
from __future__ import annotations
import hashlib,subprocess,tempfile,unittest
from pathlib import Path
from hardsub_extractor.models import Cue,OCRObservation
from hardsub_extractor.ocr.base import OCREngine
from hardsub_extractor.pipeline import extract
from hardsub_extractor.extraction_progress import ExtractionMonitor,ExtractionCancelled
from hardsub_extractor.frame_source import NormalizedROI
from hardsub_extractor.runtime_paths import resolve_binary

class ColorOCR(OCREngine):
    calls=0
    def recognize(self,frame):
        self.calls+=1
        pixel=frame.image[0,0]
        if pixel[2]>150 and pixel[0]<100:
            return OCRObservation('안녕 {세계}',.99,frame.pts)
        if pixel[0]>150 and pixel[2]<100:
            return OCRObservation('你好 \\N',.99,frame.pts)
        return OCRObservation('',0.0,frame.pts)

class PipelineWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='hse-real-wire-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.video=self.root/'source.mkv'
        try:
            ffmpeg = resolve_binary('ffmpeg')
            resolve_binary('ffprobe')
        except FileNotFoundError:
            self.skipTest('Bundled or PATH FFmpeg/FFprobe not available')
        subprocess.run([ffmpeg,'-nostdin','-loglevel','error','-y',
          '-f','lavfi','-i','color=c=red:s=64x64:r=8:d=2',
          '-f','lavfi','-i','color=c=blue:s=64x64:r=8:d=2',
          '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0',
          '-c:v','ffv1',str(self.video)],check=True,timeout=20)
        self.sha=hashlib.sha256(self.video.read_bytes()).hexdigest()

    def test_end_to_end_decode_recognize_export_and_nooverwrite(self):
        events=[];ocr=ColorOCR()
        m=ExtractionMonitor(on_progress=events.append,is_cancelled=lambda:False,exclusive_outputs=True)
        report=extract(str(self.video),ocr,self.root/'out',streaming=True,
            roi=NormalizedROI(0,0,1,1),monitor=m)
        self.assertEqual(report.cue_count,2)
        srt=report.srt_path.read_text(encoding='utf8')
        ass=report.ass_path.read_text(encoding='utf8')
        self.assertIn('안녕 {세계}',srt)
        self.assertIn('你好 \\N',srt)
        self.assertIn('\\{{}세계}',ass)
        self.assertIn('\\\u2060N',ass)
        self.assertEqual(events[-1].stage,'completed')
        self.assertEqual(hashlib.sha256(self.video.read_bytes()).hexdigest(),self.sha)
        self.assertGreater(ocr.calls,1)
        with self.assertRaises(FileExistsError):
            extract(str(self.video),ColorOCR(),self.root/'out',streaming=True,
                roi=NormalizedROI(0,0,1,1),monitor=m)
        self.assertEqual(hashlib.sha256(self.video.read_bytes()).hexdigest(),self.sha)

    def test_bad_timing_refiner_never_publishes_files(self):
        def corrupt(cues,source,roi):
            return [Cue(c.start,c.end,'WRONG',c.confidence) for c in cues]
        with self.assertRaisesRegex(ValueError,'timing_refiner_broke_subtitle_invariants'):
            extract(str(self.video),ColorOCR(),self.root/'outbad',streaming=True,
                roi=NormalizedROI(0,0,1,1),timing_refiner=corrupt,
                monitor=ExtractionMonitor(exclusive_outputs=True))
        self.assertFalse((self.root/'outbad.srt').exists())
        self.assertFalse((self.root/'outbad.ass').exists())

    def test_pre_cancel_never_publishes_files(self):
        with self.assertRaises(ExtractionCancelled):
            extract(str(self.video),ColorOCR(),self.root/'outcancel',streaming=True,
                roi=NormalizedROI(0,0,1,1),
                monitor=ExtractionMonitor(is_cancelled=lambda:True,exclusive_outputs=True))
        self.assertFalse((self.root/'outcancel.srt').exists())
        self.assertFalse((self.root/'outcancel.ass').exists())

if __name__=='__main__':unittest.main()
