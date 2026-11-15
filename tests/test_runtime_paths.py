import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hardsub_extractor.runtime_paths import resolve_binary


class RuntimePathTests(unittest.TestCase):
    def test_explicit_binary_env(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"ffmpeg-custom"
            p.write_bytes(b"x")
            with patch.dict(os.environ,{"HSE_FFMPEG":str(p)},clear=False):
                self.assertEqual(resolve_binary("ffmpeg"),str(p))

    def test_bad_explicit_path_fails_loudly(self):
        with patch.dict(os.environ,{"HSE_FFMPEG":"/definitely/missing"},clear=False):
            with self.assertRaises(FileNotFoundError):
                resolve_binary("ffmpeg")

    def test_bundled_binary_is_preferred_over_path(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            bundled=root/"bin"/("ffmpeg.exe" if os.name=="nt" else "ffmpeg")
            bundled.parent.mkdir(parents=True)
            bundled.write_bytes(b"real-bundled-binary-placeholder")
            with patch("hardsub_extractor.runtime_paths.bundle_root",return_value=root), patch(
                "hardsub_extractor.runtime_paths.shutil.which",return_value="path-ffmpeg"
            ):
                self.assertEqual(resolve_binary("ffmpeg"),str(bundled))


if __name__=="__main__":
    unittest.main()
