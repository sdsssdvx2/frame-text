"""Bounded exact-byte provenance probe for a public CC BY 3.0 animated trailer.

The source is the W3C-hosted Sintel trailer, acknowledged by the Blender
Foundation's Sintel CC BY 3.0 sharing conditions. No browser profiles,
private media, cookies, scripts, executables or DRM paths are downloaded.
Fingerprint mode does NOT authorize actual benchmarking: the first real
download must be independently pinned in code before model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

SOURCE = "https://media.w3.org/2010/05/sintel/trailer.mp4"
ATTRIBUTION = "Blender Foundation | durian.blender.org (CC BY 3.0)"
MAX_SOURCE_BYTES = 6 * 1024 * 1024
# Set only from the witnessed exact-source fingerprint. Empty is fail-closed.
EXPECTED_SOURCE_SHA256 = "b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254"


def fetch_source(dest: Path, *, required_hash: str | None) -> tuple[str, int]:
    """Bounded HTTPS file-only retrieval with a single allowed destination.

    A failed or changing source is not allowed to create a successful
    comparative-quality result. This never follows redirects to another host.
    """
    if dest.exists() or dest.is_symlink():
        raise RuntimeError("preexisting temporary source")
    request = urllib.request.Request(
        SOURCE,
        headers={"User-Agent": "HSE-Sintel-CC-BY3-independent-test/1"},
    )
    partial = dest.with_suffix(".partial")
    if partial.exists() or partial.is_symlink():
        raise RuntimeError("preexisting partial source")
    digest = hashlib.sha256()
    count = 0
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.geturl() != SOURCE:
                raise RuntimeError("unexpected video redirect")
            if response.status != 200:
                raise RuntimeError("video status not success")
            declared = response.headers.get("Content-Length")
            if declared is not None and (
                not declared.isdecimal() or int(declared) > MAX_SOURCE_BYTES
            ):
                raise RuntimeError("untrusted video content length")
            with partial.open("xb") as stream:
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > MAX_SOURCE_BYTES:
                        raise RuntimeError("video exceeded byte budget")
                    digest.update(block)
                    stream.write(block)
        sha = digest.hexdigest()
        if count < 1024 or (required_hash is not None and (
            len(required_hash) != 64 or sha != required_hash
        )):
            raise RuntimeError("Sintel trailer is empty or SHA mismatch")
        if declared is not None and count != int(declared):
            raise RuntimeError("video Content-Length changed during retrieval")
        partial.rename(dest)
        return sha, count
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def probe_duration_and_dimensions(video: Path) -> dict:
    """Read-only FFprobe with no network protocols and a bounded JSON result."""
    from hardsub_extractor.runtime_paths import resolve_binary
    command = [
        resolve_binary("ffprobe"), "-v", "error",
        "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height:format=duration",
        "-of", "json", str(video),
    ]
    proc = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False, timeout=35,
    )
    if proc.returncode or len(proc.stdout) > 32768:
        raise RuntimeError("Sintel video decode metadata probe failed")
    obj = json.loads(proc.stdout)
    streams = obj.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("unexpected number of video streams")
    v = streams[0]
    duration = float(obj["format"]["duration"])
    if (
        v.get("codec_name") not in ("h264", "mpeg4")
        or not (640 <= v.get("width", 0) <= 2048)
        or not (360 <= v.get("height", 0) <= 1080)
        or not 30 <= duration <= 90
    ):
        raise RuntimeError("Sintel original video metadata out of bounds")
    return {
        "codec": v["codec_name"], "width": v["width"],
        "height": v["height"], "duration_seconds": round(duration, 3),
    }


def probe_source(*, require_pin: bool) -> dict:
    """Fetch in a disposable location; never keep media across executions."""
    if require_pin and len(EXPECTED_SOURCE_SHA256) != 64:
        raise RuntimeError("reference source SHA has not been pinned")
    with tempfile.TemporaryDirectory(prefix="hse-sintel-source-") as td:
        original = Path(td) / "sintel-original.mp4"
        sha, size = fetch_source(
            original,
            required_hash=EXPECTED_SOURCE_SHA256 if require_pin else None,
        )
        meta = probe_duration_and_dimensions(original)
        if hashlib.sha256(original.read_bytes()).hexdigest() != sha:
            raise RuntimeError("original bytes changed after decoder probe")
        return {
            "schema": 1,
            "source": "W3C Sintel trailer, Blender Foundation",
            "license": "CC BY 3.0 attribution required",
            "original_file_sha256": sha,
            "original_bytes": size,
            "media": meta,
            "sha_pinned_and_matched": require_pin,
            "user_content_touched": False,
            "remote_code_executed": False,
            "media_retained": False,
            "human_transcription_verified": False,
            "naturally_burned_in_dialogue_verified": False,
            "film_background_benchmark_verified": False,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint-only", action="store_true")
    ap.add_argument("--require-pinned", action="store_true")
    args = ap.parse_args()
    if args.fingerprint_only == args.require_pinned:
        ap.error("choose exactly one of --fingerprint-only and --require-pinned")
    try:
        result = probe_source(require_pin=args.require_pinned)
    except Exception as exc:
        print("HSE_SINTEL_SOURCE_ERROR " + json.dumps({
            "error_class": type(exc).__name__, "successful_proof": False,
        }), flush=True)
        return 3
    print("HSE_SINTEL_SOURCE_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
