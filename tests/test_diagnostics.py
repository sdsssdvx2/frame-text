import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from hardsub_extractor.diagnostics import DiagnosticRecorder
from hardsub_extractor.models import Frame, OCRObservation, VisualSegment


class DiagnosticTests(unittest.TestCase):
    def test_records_frames_and_json(self):
        with TemporaryDirectory() as td:
            d=DiagnosticRecorder(td)
            frame=Frame(1.25,np.zeros((32,100,3),dtype=np.uint8))
            seg=VisualSegment(1.0,2.0,[frame],{"settled_by":"test"})
            obs=[OCRObservation("테스트",.9,1.25)]
            d.record_segment(0,seg,obs,obs[0])
            path=d.write(extra={"mode":"unit"})
            data=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["segments"][0]["voted"]["text"],"테스트")
            img=Path(td)/data["segments"][0]["samples"][0]["file"]
            self.assertTrue(img.is_file())


if __name__=="__main__":
    unittest.main()
