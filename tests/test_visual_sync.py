import subprocess
import sys
import unittest
from pathlib import Path

from hardsub_extractor.models import Cue
from hardsub_extractor.visual_sync import (
    apply_global_offset,
    estimate_global_offset,
)


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"


class VisualSyncTests(unittest.TestCase):
    def fixture(self):
        src=ART/"sync-source.mp4"
        dst=ART/"sync-target-plus-0750ms.mp4"
        if not src.exists() or not dst.exists():
            subprocess.run([sys.executable,str(ROOT/"tools/make_sync_fixture.py")],check=True)
        return src,dst

    def test_global_visual_offset_with_burned_subtitles(self):
        src,dst=self.fixture()
        result=estimate_global_offset(
            str(src),str(dst),
            sample_fps=4.0,
            max_abs_offset=2.0,
        )
        self.assertAlmostEqual(result.offset,0.75,delta=0.25)
        self.assertGreater(result.score,0.78)
        self.assertGreater(result.matched_frames,20)

    def test_apply_offset(self):
        cues=[Cue(1.0,2.0,"테스트")]
        got=apply_global_offset(cues,.75)
        self.assertEqual(got[0].start,1.75)
        self.assertEqual(got[0].end,2.75)


if __name__=="__main__":unittest.main()
