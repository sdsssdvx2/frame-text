import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hardsub_extractor.models import Cue
from hardsub_extractor.writers import write_srt
from hardsub_extractor.sync_workflow import sync_srt_to_target


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"


class SyncWorkflowTests(unittest.TestCase):
    def fixture(self):
        a=ART/"piecewise-source.mp4"
        b=ART/"piecewise-target.mp4"
        if not a.exists() or not b.exists():
            subprocess.run(
                [sys.executable,str(ROOT/"tools/make_piecewise_sync_fixture.py")],
                check=True,
            )
        return a,b

    def test_srt_to_piecewise_synced_outputs(self):
        a,b=self.fixture()
        with TemporaryDirectory() as td:
            source=Path(td)/"source.srt"
            write_srt(
                [Cue(2.0,3.0,"앞"),Cue(10.0,11.0,"뒤")],
                source,
            )
            r=sync_srt_to_target(
                source,str(a),str(b),Path(td)/"final",
                sample_fps=4.0,
                window_seconds=3.0,
                anchor_spacing=1.5,
                max_abs_offset=2.0,
            )
            text=r.srt_path.read_text(encoding="utf-8")
            self.assertIn("00:00:02,750 --> 00:00:03,750",text)
            self.assertIn("00:00:11,250 --> 00:00:12,250",text)
            self.assertTrue(r.ass_path.is_file())
            self.assertTrue(r.report_path.is_file())


if __name__=="__main__":
    unittest.main()
