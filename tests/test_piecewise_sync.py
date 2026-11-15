import subprocess,sys,unittest
from pathlib import Path

from hardsub_extractor.models import Cue
from hardsub_extractor.visual_sync import (
    apply_piecewise_offsets,
    estimate_piecewise_offsets,
    offset_at_time,
)

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"

class PiecewiseSyncTests(unittest.TestCase):
    def fixture(self):
        a=ART/"piecewise-source.mp4";b=ART/"piecewise-target.mp4"
        if not a.exists() or not b.exists():
            subprocess.run([sys.executable,str(ROOT/"tools/make_piecewise_sync_fixture.py")],check=True)
        return a,b

    def test_detects_mid_episode_offset_change(self):
        a,b=self.fixture()
        r=estimate_piecewise_offsets(
            str(a),str(b),
            sample_fps=4.0,
            window_seconds=3.0,
            anchor_spacing=1.5,
            max_abs_offset=2.0,
        )
        self.assertGreaterEqual(len(r.segments),2)
        self.assertAlmostEqual(offset_at_time(r.segments,3.0),.75,delta=.25)
        self.assertAlmostEqual(offset_at_time(r.segments,11.0),1.25,delta=.25)

        cues=[Cue(2.0,3.0,"앞"),Cue(10.0,11.0,"뒤")]
        mapped=apply_piecewise_offsets(cues,r.segments)
        self.assertAlmostEqual(mapped[0].start,2.75,delta=.25)
        self.assertAlmostEqual(mapped[1].start,11.25,delta=.25)

if __name__=="__main__":unittest.main()
