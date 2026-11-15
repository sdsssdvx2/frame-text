from __future__ import annotations
import json, math, queue, re, subprocess, threading, time
from dataclasses import dataclass
from typing import Callable, Iterator
import numpy as np
from .models import Frame
from .runtime_paths import resolve_binary

_SHOWINFO_RE = re.compile(r"showinfo.*?\bn:\s*(\d+).*?\bpts_time:([-+]?\d+(?:\.\d+)?)")

@dataclass(slots=True)
class VideoInfo:
    width: int
    height: int
    avg_frame_rate: float
    duration: float | None

@dataclass(slots=True)
class NormalizedROI:
    x: float = 0.04
    y: float = 0.58
    w: float = 0.92
    h: float = 0.40
    def pixels(self, width: int, height: int) -> tuple[int,int,int,int]:
        # Most Anime/HLS inputs are 4:2:0. FFmpeg's crop filter rounds
        # chroma-subsampled coordinates/sizes unless exact=1, so normalize
        # to even pixels and keep our raw-frame byte accounting exact.
        def even_down(v: int) -> int:
            return max(0, v - (v % 2))
        x = even_down(max(0, min(width - 2, round(width * self.x))))
        y = even_down(max(0, min(height - 2, round(height * self.y))))
        w = even_down(max(2, min(width - x, round(width * self.w))))
        h = even_down(max(2, min(height - y, round(height * self.h))))
        if x + w > width:
            w = even_down(width - x)
        if y + h > height:
            h = even_down(height - y)
        return int(x), int(y), int(w), int(h)

def _rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        a,b=value.split("/",1)
        return float(a)/float(b) if float(b) else 0.0
    return float(value)


def _build_ffprobe_command(
    source: str,
    *,
    extra_input_args: list[str] | None = None,
) -> list[str]:
    """Build the exact ffprobe argv used for an input.

    Keeping input options in a pure builder lets CI verify header/session option
    placement without depending on a localhost HTTP stack that can be modified
    by software outside HSE on a self-hosted Windows runner.
    """
    return [
        resolve_binary("ffprobe"),
        "-v", "error",
        *(extra_input_args or []),
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json",
        source,
    ]


def _build_ffmpeg_decode_command(
    source: str,
    *,
    extra_input_args: list[str] | None = None,
    start_seconds: float = 0.0,
    max_duration_seconds: float | None = None,
    video_filter: str,
) -> list[str]:
    """Build the exact ffmpeg raw-frame decode argv used by the frame source."""
    return [
        resolve_binary("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "info",
        *(extra_input_args or []),
        *(["-ss", str(start_seconds)] if start_seconds else []),
        "-i", source,
        *(["-t", str(max_duration_seconds)] if max_duration_seconds is not None else []),
        "-map", "0:v:0", "-an", "-sn",
        "-vf", video_filter,
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]


def probe_video(
    source: str,
    *,
    extra_input_args: list[str] | None = None,
    timeout_seconds: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> VideoInfo:
    # Preserve legacy behavior for all existing extractors. Only opt-in
    # read-only preview calls bound the metadata probe to avoid a stuck GUI.
    extra_run_args = {}
    if timeout_seconds is not None:
        if (not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 120):
            raise ValueError("invalid FFprobe timeout")
        extra_run_args["timeout"] = timeout_seconds
    cmd = _build_ffprobe_command(source, extra_input_args=extra_input_args)
    if cancel_requested is None:
        # Keep ordinary unsupervised extraction exactly as it was.
        p = subprocess.run(cmd, capture_output=True, text=True, check=True,
                           **extra_run_args)
        stdout = p.stdout
    else:
        if not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable")
        from .extraction_progress import ExtractionCancelled
        if cancel_requested():
            raise ExtractionCancelled("FFprobe cancelled before launch")
        # subprocess.run(timeout=...) cannot respond to a GUI cancel until
        # ffprobe exits or times out. Supervise ONLY the child started here.
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        began = time.monotonic()
        try:
            while True:
                if cancel_requested():
                    raise ExtractionCancelled("FFprobe cancelled")
                remaining = (
                    timeout_seconds - (time.monotonic() - began)
                    if timeout_seconds is not None else None
                )
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                try:
                    stdout, stderr = proc.communicate(
                        timeout=min(.1, remaining) if remaining is not None else .1,
                    )
                except subprocess.TimeoutExpired:
                    continue
                if cancel_requested():
                    raise ExtractionCancelled("FFprobe cancelled")
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode, cmd, output=stdout, stderr=stderr,
                    )
                break
        finally:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except (OSError, ProcessLookupError):
                    pass
                try:
                    proc.communicate(timeout=1.5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        proc.communicate(timeout=2.0)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError(
                            "FFprobe cleanup could not confirm process exit"
                        ) from exc
            # communicate() closes both pipes on completion. A cancel just
            # after child exit may bypass communicate(): close those too.
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
    root=json.loads(stdout)
    stream=root["streams"][0]
    d=root.get("format",{}).get("duration")
    return VideoInfo(int(stream["width"]),int(stream["height"]),_rate(stream.get("avg_frame_rate")),
                     float(d) if d not in (None,"N/A") else None)

class FFmpegPTSFrameSource:
    def __init__(
        self,
        source: str,
        roi: NormalizedROI | None = None,
        *,
        extra_input_args: list[str] | None = None,
        sample_fps: float | None = None,
        max_duration_seconds: float | None = None,
        start_seconds: float = 0.0,
        preprobed_info: VideoInfo | None = None,
        decode_timeout_seconds: float | None = None,
        probe_timeout_seconds: float | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ):
        # Optional preview-only watchdog. None preserves the existing
        # full-episode streaming contract with zero monitor threads.
        if decode_timeout_seconds is not None and (
            not math.isfinite(decode_timeout_seconds)
            or not 0 < decode_timeout_seconds <= 120
        ):
            raise ValueError("invalid FFmpeg decode timeout")
        if probe_timeout_seconds is not None and (
            not math.isfinite(probe_timeout_seconds)
            or not 0 < probe_timeout_seconds <= 120
        ):
            raise ValueError("invalid FFprobe timeout")
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable")
        self.decode_timeout_seconds = decode_timeout_seconds
        self.probe_timeout_seconds = probe_timeout_seconds
        self.cancel_requested = cancel_requested
        # Only an explicit read-only caller that already probed this exact
        # source supplies metadata. Normal extraction retains its own probe.
        self.preprobed_info = preprobed_info
        self.source = source
        self.roi = roi or NormalizedROI()
        self.extra_input_args = list(extra_input_args or [])
        self.sample_fps = sample_fps
        if max_duration_seconds is not None and (
            not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be finite and non-negative")
        self.max_duration_seconds = max_duration_seconds
        self.start_seconds = float(start_seconds)

    def __iter__(self) -> Iterator[Frame]:
        info = self.preprobed_info
        if info is None:
            info = probe_video(
                self.source,
                extra_input_args=self.extra_input_args,
                **({"timeout_seconds": self.probe_timeout_seconds}
                   if self.probe_timeout_seconds is not None else {}),
                **({"cancel_requested": self.cancel_requested}
                   if self.cancel_requested is not None else {}),
            )
        if self.cancel_requested is not None and self.cancel_requested():
            from .extraction_progress import ExtractionCancelled
            raise ExtractionCancelled("preview decoder cancelled")
        x,y,w,h=self.roi.pixels(info.width,info.height)
        vf = f"crop={w}:{h}:{x}:{y}"
        if self.sample_fps:
            if self.sample_fps <= 0:
                raise ValueError("sample_fps must be positive")
            vf += f",fps={self.sample_fps}"
        vf += ",showinfo"

        cmd = _build_ffmpeg_decode_command(
            self.source,
            extra_input_args=self.extra_input_args,
            start_seconds=self.start_seconds,
            max_duration_seconds=self.max_duration_seconds,
            video_filter=vf,
        )
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=10**7)
        assert proc.stdout and proc.stderr
        ptsq: queue.Queue[float]=queue.Queue()
        tail=[]

        def stderr_reader():
            for raw in iter(proc.stderr.readline,b""):
                line=raw.decode("utf-8","replace")
                tail.append(line.rstrip())
                if len(tail)>80: del tail[:-80]
                m=_SHOWINFO_RE.search(line)
                if m: ptsq.put(float(m.group(2)))

        th=threading.Thread(target=stderr_reader,daemon=True)
        th.start()
        # The stdout pipe can block indefinitely inside read() if a damaged
        # local media stream stalls FFmpeg. Only callers explicitly opting in
        # get this bounded process supervisor; normal extraction is unchanged.
        watchdog_done = threading.Event()
        watchdog_timed_out = threading.Event()
        watchdog_cancelled = threading.Event()
        watchdog = None
        if self.decode_timeout_seconds is not None or self.cancel_requested is not None:
            def watch_decoder():
                expires = (
                    time.monotonic() + self.decode_timeout_seconds
                    if self.decode_timeout_seconds is not None else None
                )
                while not watchdog_done.wait(0.05):
                    if self.cancel_requested is not None and self.cancel_requested():
                        watchdog_cancelled.set()
                    elif expires is not None and time.monotonic() >= expires:
                        watchdog_timed_out.set()
                    else:
                        continue
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                        except (OSError, ProcessLookupError):
                            pass
                        # If graceful exit leaves stdout.read blocked, force
                        # closure. The generator still owns wait()/reaping.
                        if not watchdog_done.wait(1.0) and proc.poll() is None:
                            try:
                                proc.kill()
                            except (OSError, ProcessLookupError):
                                pass
                    return
            watchdog = threading.Thread(target=watch_decoder, daemon=True)
            watchdog.start()
        size=w*h*3
        reached_eof=False
        try:
            while True:
                buf=bytearray()
                while len(buf)<size:
                    chunk=proc.stdout.read(size-len(buf))
                    if not chunk: break
                    buf.extend(chunk)
                if not buf:
                    reached_eof=True
                    break
                if len(buf)!=size:
                    raise RuntimeError(f"FFmpeg raw frame truncated: {len(buf)} / {size}")
                if watchdog is None:
                    # Preserve the exact, previously Windows-tested full
                    # extraction path when preview supervision is not enabled.
                    try:
                        pts = ptsq.get(timeout=10)
                    except queue.Empty as exc:
                        raise RuntimeError(
                            "FFmpeg frame arrived without showinfo PTS"
                        ) from exc
                else:
                    # Poll only the opt-in preview PTS wait so cancellation
                    # and the decoder time limit can interrupt promptly.
                    pts_deadline = time.monotonic() + 10.0
                    while True:
                        if watchdog_cancelled.is_set():
                            from .extraction_progress import ExtractionCancelled
                            raise ExtractionCancelled("preview decoder cancelled")
                        if watchdog_timed_out.is_set():
                            raise RuntimeError("preview decoder exceeded time limit")
                        remaining = pts_deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError(
                                "FFmpeg frame arrived without showinfo PTS"
                            )
                        try:
                            pts = ptsq.get(timeout=min(remaining, 0.1))
                            break
                        except queue.Empty:
                            continue
                img=np.frombuffer(buf,dtype=np.uint8).reshape((h,w,3)).copy()
                yield Frame(pts,img)
        finally:
            # Stop the watchdog before cleanup; only this generator owns
            # process wait and error reporting. Already-triggered abort state
            # remains readable after the watcher exits.
            watchdog_done.set()
            if watchdog is not None:
                watchdog.join(timeout=2)
            # If the consumer stopped early (GUI cancel, exception or
            # generator.close()), do not wait for FFmpeg to decode the rest of
            # the video and do not report its termination as a decode error.
            # The real EOF path still checks the exit code strictly.
            if not reached_eof and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            try:
                proc.stdout.close()
            except OSError:
                pass
            wait_error=None
            rc=None
            try:
                rc=proc.wait(timeout=10 if reached_eof else 3)
            except subprocess.TimeoutExpired as exc:
                wait_error=exc
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    rc=proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                th.join(timeout=2)
                try:
                    proc.stderr.close()
                except OSError:
                    pass
            if rc is None:
                # Never report clean cancellation while a decoder process is
                # still unconfirmed after terminate -> kill -> second wait.
                raise RuntimeError("FFmpeg cleanup could not confirm process exit")
            if watchdog_cancelled.is_set():
                from .extraction_progress import ExtractionCancelled
                raise ExtractionCancelled("preview decoder cancelled")
            if watchdog_timed_out.is_set():
                raise RuntimeError("preview decoder exceeded time limit")
            if reached_eof:
                if wait_error is not None:
                    raise RuntimeError("FFmpeg did not exit after raw-frame EOF") from wait_error
                if rc!=0:
                    raise RuntimeError(f"FFmpeg decode failed ({rc})\n"+"\n".join(tail[-20:]))
