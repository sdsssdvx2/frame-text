import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from hardsub_extractor.frame_source import FFmpegPTSFrameSource, NormalizedROI
from hardsub_extractor.runtime_paths import resolve_binary


ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"


class HeaderHLSHandler(BaseHTTPRequestHandler):
    root: Path | None = None
    expected_ua = "HSE-Test-UA/1.0"
    expected_referer = ""
    requests = 0
    rejects = 0

    def log_message(self,*_):
        pass

    def do_GET(self):
        type(self).requests += 1
        ua=self.headers.get("User-Agent","")
        ref=self.headers.get("Referer","")
        if ua != self.expected_ua or ref != self.expected_referer:
            type(self).rejects += 1
            self.send_response(403)
            self.end_headers()
            return

        rel=urlparse(self.path).path.lstrip("/")
        path=(self.root/rel) if self.root else None
        if not path or not path.is_file():
            self.send_response(404)
            self.end_headers()
            return

        data=path.read_bytes()
        if path.suffix==".m3u8":
            ctype="application/vnd.apple.mpegurl"
        elif path.suffix==".ts":
            ctype="video/mp2t"
        else:
            ctype="application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type",ctype)
        self.send_header("Content-Length",str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HLSHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=TemporaryDirectory()
        cls.hls=Path(cls.tmp.name)

        video=ART/"synthetic-korean-hardsub.mp4"
        truth=ART/"synthetic-truth.json"
        if not video.exists() or not truth.exists():
            subprocess.run(
                [sys.executable,str(ROOT/"tools/make_synthetic_fixture.py")],
                check=True,
            )

        # MPEG-2 video is native to MPEG-TS and available in the pinned LGPL
        # FFmpeg build. MPEG-4 Part 2 inside TS can lose dimensions/extradata
        # across segment boundaries, which would make this HTTP-header test
        # accidentally test a fragile container/codec combination instead.
        subprocess.run([
            resolve_binary("ffmpeg"),
            "-y","-hide_banner","-loglevel","error",
            "-i",str(video),
            "-an",
            "-c:v","mpeg2video",
            "-q:v","2",
            "-g","15",
            "-bf","0",
            "-hls_time","1",
            "-hls_list_size","0",
            "-hls_segment_filename",str(cls.hls/"seg-%03d.ts"),
            str(cls.hls/"stream.m3u8"),
        ],check=True)

        HeaderHLSHandler.root=cls.hls
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),HeaderHLSHandler)
        port=cls.server.server_port
        HeaderHLSHandler.expected_referer=f"http://127.0.0.1:{port}/player"
        HeaderHLSHandler.requests=0
        HeaderHLSHandler.rejects=0
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f"http://127.0.0.1:{port}/stream.m3u8"
        cls.referer=HeaderHLSHandler.expected_referer

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def test_probe_and_decode_both_forward_headers(self):
        frames=list(
            FFmpegPTSFrameSource(
                self.url,
                roi=NormalizedROI(),
                extra_input_args=[
                    "-user_agent",HeaderHLSHandler.expected_ua,
                    "-referer",self.referer,
                ],
                sample_fps=2.0,
            )
        )
        self.assertGreater(len(frames),5)
        self.assertGreater(HeaderHLSHandler.requests,2)
        self.assertEqual(HeaderHLSHandler.rejects,0)


if __name__=="__main__":
    unittest.main()
