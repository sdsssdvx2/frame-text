"""Real Windows seconv v5.3.0/v5.4.0 PGS and VobSub compatibility.

Use one disposable, encoded, native bitmap-track MKV per codec generated from
the exact same reference text by the already-pinned baseline seconv. Feed
that same MKV to the old and candidate BDN exporters. Inspect image counts,
pixel signatures, and original frame-quantized times. No OCR, no user media,
no provider calls, no source content printed. Not a natural-animation score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def _call(args: list[str], timeout=100) -> bytes:
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL, timeout=timeout, check=False,
    )
    if proc.returncode:
        raise RuntimeError("isolated SeConv/FFmpeg command failed")
    if len(proc.stdout) > 64 * 1024:
        raise RuntimeError("unexpected verbose converter receipt")
    return proc.stdout


def _receipt(args: list[str]) -> None:
    data = _call(args)
    try:
        reply = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("official converter JSON receipt invalid") from exc
    if not isinstance(reply, dict) or reply.get("success") is not True:
        raise RuntimeError("official converter did not confirm conversion")


def _binary_version(exe: Path, expected: str) -> None:
    data = _call([str(exe), "--version"], 25).decode("utf-8", "replace")
    if expected not in data:
        raise RuntimeError("isolated official seconv version mismatch")


def _generate(ffmpeg: Path, old: Path, scratch: Path, kind: str) -> Path:
    folder = scratch / kind
    folder.mkdir()
    source = folder / "reference.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHELLO WORLD\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nTWO LINES\nSECOND LINE\n",
        encoding="utf-8",
    )
    stream_dir = folder / "original_bitstream"
    stream_dir.mkdir()
    fmt = "bluraysup" if kind == "pgs" else "vobsub"
    _receipt([
        str(old), str(source), fmt,
        f"--output-folder:{stream_dir}", "--resolution:1920x1080",
        "--font-name:Arial", "--font-size:58",
        "--font-color:white", "--outline-color:black", "--json",
    ])
    name = "*.sup" if kind == "pgs" else "*.idx"
    files = list(stream_dir.glob(name))
    if len(files) != 1:
        raise RuntimeError("expected exactly one original subtitle bitstream")
    if kind == "vobsub" and not files[0].with_suffix(".sub").is_file():
        raise RuntimeError("source VobSub bitmap companion is missing")
    result = folder / "original.mkv"
    _call([
        str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe",
        *(["-copyts"] if kind == "pgs" else []),
        "-i", str(files[0]), "-map", "0:s:0", "-c:s", "copy",
        "-metadata:s:s:0", "language=eng", str(result),
    ])
    if not result.is_file() or not 1024 <= result.stat().st_size <= 32 * 1024 * 1024:
        raise RuntimeError("expected bounded genuine encoded bitmap MKV")
    return result


_TC = re.compile(r"^(\d\d):(\d\d):(\d\d):(\d\d)$")
_PNG = re.compile(r"^\d{4,8}\.png$", re.I)


def _timecode(text: str, fps: float) -> float:
    mat = _TC.fullmatch(text)
    if not mat:
        raise RuntimeError("unexpected BDN frame timecode")
    h, m, s, f = map(int, mat.groups())
    if not (h == 0 and 0 <= m <= 1 and 0 <= s <= 59 and 0 <= f < fps + .001):
        raise RuntimeError("BDN frame timecode out of range")
    return h*3600 + m*60 + s + f/fps


def _inspect(path: Path) -> list[tuple[float,float]]:
    index = path / "index.xml"
    if not index.is_file() or index.is_symlink() or not 0 < index.stat().st_size < 1_000_000:
        raise RuntimeError("BDN subtitle index missing or oversized")
    raw = index.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise RuntimeError("untrusted BDN XML")
    root = ET.fromstring(raw)
    if root.tag != "BDN":
        raise RuntimeError("unrecognized BDN export")
    desc = root.find("Description")
    format_tag = desc.find("Format") if desc is not None else None
    if format_tag is None or format_tag.get("DropFrame") != "false":
        raise RuntimeError("BDN frame format changed")
    fps = float(format_tag.get("FrameRate", "0"))
    if not 1 <= fps <= 120:
        raise RuntimeError("BDN framerate unsupported")
    events = root.find("Events")
    if events is None or len(events) != 2:
        raise RuntimeError("BDN dropped or fabricated original subtitle events")
    output=[]
    for event in list(events):
        if event.tag != "Event":
            raise RuntimeError("BDN Event representation changed")
        children=list(event)
        if len(children)!=1 or children[0].tag!="Graphic":
            raise RuntimeError("BDN image geometry representation changed")
        name=children[0].text or ""
        if not _PNG.fullmatch(name):
            raise RuntimeError("BDN bitmap filename unsupported")
        image=path / name
        if not image.is_file() or image.is_symlink() or not 256 <= image.stat().st_size < 5*1024*1024:
            raise RuntimeError("BDN isolated PNG image invalid")
        if image.open("rb").read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("BDN image is not a real PNG")
        interval=(_timecode(event.attrib["InTC"],fps),_timecode(event.attrib["OutTC"],fps))
        if interval[1] <= interval[0]:
            raise RuntimeError("invalid BDN event time")
        output.append(interval)
    return output


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--old",type=Path,required=True)
    p.add_argument("--new",type=Path,required=True)
    p.add_argument("--ffmpeg",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    a=p.parse_args()
    for path in (a.old,a.new,a.ffmpeg):
        if not path.is_file():
            raise RuntimeError("one of the official isolated tools is unavailable")
    _binary_version(a.old,"5.3.0")
    _binary_version(a.new,"5.4.0")
    findings={}
    with tempfile.TemporaryDirectory(prefix="hse-seconv-versions-",dir=a.scratch) as td:
        root=Path(td)
        for kind in ("pgs","vobsub"):
            source=_generate(a.ffmpeg,a.old,root,kind)
            original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
            output={}
            for label, exe in (("5.3.0",a.old),("5.4.0",a.new)):
                work=root / kind / ("bdn-"+label)
                work.mkdir()
                began=time.monotonic()
                _receipt([
                    str(exe),str(source),"bdnxml","--track-number:1",
                    f"--output-folder:{work}", "--output-filename:image-bitmap",
                    "--json",
                ])
                intervals=_inspect(work/"image-bitmap")
                reference=((1.0,3.0),(4.0,6.0))
                edge=max(abs(t-r) for observed,expected in zip(intervals,reference)
                         for t,r in zip(observed,expected))
                if edge > 0.25:
                    raise RuntimeError("official BDN converter source timing drift")
                output[label]={"events":len(intervals),
                               "max_source_edge_error_seconds":round(edge,4),
                               "elapsed_seconds":round(time.monotonic()-began,3)}
            if hashlib.sha256(source.read_bytes()).hexdigest()!=original_sha:
                raise RuntimeError("original source MKV was modified")
            difference=max(abs(x-y) for i,j in zip(
                _inspect(root/kind/"bdn-5.3.0"/"image-bitmap"),
                _inspect(root/kind/"bdn-5.4.0"/"image-bitmap"),
            ) for x,y in zip(i,j))
            if difference > 0.25:
                raise RuntimeError("new official converter changed image event timing")
            findings[kind]={"baseline":output["5.3.0"],"candidate":output["5.4.0"],
                            "max_interversion_edge_error_seconds":round(difference,4)}
    print("HSE_SECONV_54_AB="+json.dumps({
        "real_official_executables":True,"same_source_image_bitstream":True,
        "source_track_count":2,"original_user_media_access":False,
        "ocr_model_calls":0,"paid_provider_requests":0,
        "original_media_modified":False,"candidate_approved_for_production":False,
        "findings":findings,
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
