from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Callable
import statistics
import math
import cv2, numpy as np
from .models import Frame, VisualSegment

@dataclass(slots=True)
class DetectorConfig:
    signature_width: int = 320

    # Absolute floors. The real boundary threshold is adaptive:
    # max(floor, rolling_background_median * ratio + margin).
    change_threshold: float = 0.010
    stable_threshold: float = 0.006
    adaptive_change_ratio: float = 1.45
    adaptive_change_margin: float = 0.002
    adaptive_stable_ratio: float = 1.25
    adaptive_stable_margin: float = 0.003
    baseline_window: int = 21
    baseline_warmup_frames: int = 6
    regime_stable_frames: int = 5
    regime_spread_abs: float = 0.004
    regime_spread_ratio: float = 0.30

    stable_frames: int = 2
    max_samples_per_segment: int = 3
    sample_spacing: float = 0.18
    # Research-only optional cap; the production default is None, preserving
    # every existing visual segment and its three-sample OCR budget.
    max_quiet_segment_seconds: float | None = None
    # Optional research guard against pending scene changes that never settle.
    # A pending transition can otherwise block the quiet cap for >1 second
    # and cause a real caption change to be detected long after it appears.
    # None is inert; never enable by default without blank/video QA.
    max_pending_change_seconds: float | None = None
    # Append the opt-in field to preserve positional DetectorConfig callers.
    signature_mode: str = "edges"

def subtitle_signature(image: np.ndarray, width: int=320) -> np.ndarray:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if gray.shape[1]!=width:
        scale=width/gray.shape[1]
        h=max(16,round(gray.shape[0]*scale))
        gray=cv2.resize(gray,(width,h),interpolation=cv2.INTER_AREA)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    edges=cv2.Canny(gray,70,150)
    edges=cv2.dilate(edges,np.ones((1,2),np.uint8),iterations=1)
    return edges>0

def signature_change(a: np.ndarray,b: np.ndarray)->float:
    return float(np.mean(np.logical_xor(a,b)))

def persistent_white_signature(*, band_start: float = 0.10, band_end: float = 0.65):
    """Stateful, stream-local near-white caption presence over 3 frames.

    Research-only opt-in for persistent bright text. Dark and yellow text are
    NOT supported by this detector. Instantiate once per decoded stream,
    never share the short history across users, videos or threads.
    Optional spatial band is research-only: the original .10..65 crop is
    unchanged unless this factory is explicitly called with new bounds.
    """
    if (
        type(band_start) is not float or type(band_end) is not float
        or not math.isfinite(band_start) or not math.isfinite(band_end)
        or not 0.0 <= band_start < band_end <= 1.0
        or band_end - band_start < 0.15
    ):
        raise ValueError("invalid_opt_in_caption_signature_band")
    recent = deque(maxlen=3)
    broad_blocks = np.ones((9, 9), dtype=np.uint8)
    thin_strokes = np.ones((1, 2), dtype=np.uint8)

    def signature(image: np.ndarray, width: int = 320) -> np.ndarray:
        h = image.shape[0]
        y0 = int(round(h * band_start))
        y1 = int(round(h * band_end))
        crop = image[y0:y1]
        if crop.shape[0] < 16:
            raise ValueError("persistent_caption_band_too_small")
        lowest = np.min(crop, axis=2)
        chroma = np.max(crop, axis=2).astype(np.int16) - lowest.astype(np.int16)
        white = np.logical_and(lowest >= 205, chroma <= 36).astype(np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_TOPHAT,
                                 broad_blocks).astype(np.float32)
        out_height = max(16, round(white.shape[0] * width / white.shape[1]))
        small = cv2.resize(white, (width, out_height),
                           interpolation=cv2.INTER_AREA) >= 0.12
        small = cv2.dilate(small.astype(np.uint8), thin_strokes,
                           iterations=1).astype(bool)
        recent.append(small)
        if len(recent) < 3:
            return np.zeros_like(small)
        return np.logical_and.reduce(tuple(recent))

    return signature


def _append_sample(samples:list[Frame], frame:Frame, cfg:DetectorConfig)->None:
    if not samples:
        samples.append(frame)
        return

    limit=max(1,int(cfg.max_samples_per_segment))
    if len(samples)<limit:
        if frame.pts-samples[-1].pts>=cfg.sample_spacing:
            samples.append(frame)
        return

    if limit==1:
        samples[0]=frame
        return
    if limit==2:
        # Preserve the first sample and always keep the true live tail.
        samples[-1]=frame
        return

    if limit==3:
        # Long quiet segments used to become [first, very-early, latest]. If
        # a subtitle changed late inside such a segment, the two stale early
        # OCR readings could outvote the new tail reading. Preserve the first
        # sample as an anchor, but roll the other two slots forward: the middle
        # slot advances whenever the previous latest sample is sufficiently
        # separated from it, while the final slot always follows the real tail.
        # This keeps the same three-call OCR budget and gives a persistent late
        # transition two chances to be observed.
        previous_latest=samples[-1]
        if previous_latest.pts-samples[-2].pts>=cfg.sample_spacing:
            samples[-2]=previous_latest
        samples[-1]=frame
        return

    # For explicitly requested larger sampling budgets, preserve the
    # first frame and roll all remaining slots forward. Previously only
    # the final slot moved at limits >3, leaving stale early observations
    # occupying the other slots. This made comparative 3/5/7 OCR studies
    # misleading and starved later subtitles of corroborating samples.
    #
    # The released three-slot default above is deliberately unchanged.
    # Keep fixed O(limit) images and maintain temporal order, even when
    # several video frames arrive faster than the configured spacing.
    previous_latest = samples[-1]
    if previous_latest.pts - samples[-2].pts >= cfg.sample_spacing:
        samples[1:-1] = samples[2:]
    samples[-1] = frame

def _median(values: deque[float]) -> float:
    return statistics.median(values) if values else 0.0

def _change_limit(cfg:DetectorConfig, baseline:float)->float:
    return max(cfg.change_threshold, baseline*cfg.adaptive_change_ratio + cfg.adaptive_change_margin)

def _stable_limit(cfg:DetectorConfig, baseline:float)->float:
    return max(cfg.stable_threshold, baseline*cfg.adaptive_stable_ratio + cfg.adaptive_stable_margin)

def iter_segment_frames(frames, cfg: DetectorConfig | None = None, *, trace: Callable[[dict], None] | None = None) -> Iterator[VisualSegment]:
    """Incrementally yield the exact same boundary and sample decisions.

    Unlike segment_frames, this does not retain every sampled image for the
    duration of a long video. The caller must close an interrupted iterator.
    """
    cfg=cfg or DetectorConfig()
    if type(cfg.signature_mode) is not str or cfg.signature_mode not in (
        "edges", "persistent_white_3", "caption_band_white_3"
    ):
        raise ValueError("invalid_caption_signature_mode")
    # Keep the *OCR frame* unchanged. This separate evidence-only band
    # matches the lower subtitle geometry of an independently authored
    # real captioned/clean video pair; it must not become a universal
    # default because raised/dark/colored captions can fall outside it.
    if cfg.signature_mode == "edges":
        signature_fn = subtitle_signature
    elif cfg.signature_mode == "persistent_white_3":
        signature_fn = persistent_white_signature()
    else:
        signature_fn = persistent_white_signature(
            band_start=0.38, band_end=0.90,
        )
    cap = cfg.max_quiet_segment_seconds
    if cap is not None and (
        isinstance(cap, bool) or not isinstance(cap, (float, int))
        or not math.isfinite(cap) or not 0.50 <= cap <= 30.0
    ):
        raise ValueError("quiet segment cap must be finite and 0.50..30.0 seconds")
    pending_timeout = cfg.max_pending_change_seconds
    if pending_timeout is not None and (
        isinstance(pending_timeout, bool)
        or not isinstance(pending_timeout, (float, int))
        or not math.isfinite(pending_timeout)
        or not 0.25 <= pending_timeout <= 5.0
    ):
        raise ValueError(
            "pending change timeout must be finite and 0.25..5.0 seconds"
        )
    it=iter(frames)
    try:
        first=next(it)
    except StopIteration:
        return

    prev=signature_fn(first.image,cfg.signature_width)
    current_start=first.pts
    current_samples=[first]
    last_pts=first.pts
    frame_deltas=deque(maxlen=60)

    # Pairwise visual change during normal motion. A subtitle boundary is
    # treated as an outlier relative to this local background-motion baseline.
    baseline_scores: deque[float]=deque(maxlen=cfg.baseline_window)

    pending_start:float|None=None
    pending_samples:list[Frame]=[]
    pending_low_scores:list[float]=[]
    pending_all_scores:list[float]=[]
    stable_count=0
    pending_cap_reported=False

    def record_state(kind: str, at: float, **extras) -> None:
        # Explicit opt-in only. No image, OCR text, title or source path.
        # The existing production decoder calls this with trace=None.
        if trace is not None:
            trace({
                "kind": kind,
                "at_pts": round(float(at), 3),
                "current_start": round(float(current_start), 3),
                "pending_start": (
                    round(float(pending_start), 3)
                    if pending_start is not None else None
                ),
                **extras,
            })

    for frame in it:
        if frame.pts>last_pts:
            frame_deltas.append(frame.pts-last_pts)

        sig=signature_fn(frame.image,cfg.signature_width)
        score=signature_change(prev,sig)
        baseline=_median(baseline_scores)

        if cap is not None and frame.pts-current_start >= cap:
            if pending_start is not None and not pending_cap_reported:
                record_state("cap_blocked_by_pending", frame.pts)
                pending_cap_reported=True
            elif (
                pending_start is None and
                len(baseline_scores) >= cfg.baseline_warmup_frames and
                score >= _change_limit(cfg, baseline)
            ):
                record_state("cap_deferred_by_new_change", frame.pts)

        # A moving background may hide a real caption onset from the
        # adaptive change detector. An explicitly requested, finite quiet
        # cap supplies a bounded fallback analysis window. Only cap when
        # there is NO pending visual-change event: scene boundaries
        # continue to take priority. The frame at the cap becomes the
        # NEXT segment's first sample; no duplicated/shared frame or
        # reference subtitle timestamp is used. Default None is inert.
        if (
            cap is not None and pending_start is None
            and frame.pts-current_start >= cap
            and (
                len(baseline_scores) < cfg.baseline_warmup_frames
                or score < _change_limit(cfg, baseline)
            )
        ):
            if frame.pts > current_start:
                record_state("quiet_cap", frame.pts)
                yield VisualSegment(
                    current_start, frame.pts, list(current_samples),
                    {"quiet_duration_cap": True,
                     "background_median": baseline},
                )
            current_start=frame.pts
            current_samples=[frame]
            baseline_scores.append(score)
            prev=sig
            last_pts=frame.pts
            continue

        if pending_start is None:
            # Before the rolling baseline exists, learn normal scene motion
            # instead of treating the very first moving frame as a subtitle change.
            if len(baseline_scores) < cfg.baseline_warmup_frames:
                baseline_scores.append(score)
                _append_sample(current_samples,frame,cfg)
            else:
                limit=_change_limit(cfg,baseline)
                if score>=limit:
                    pending_start=frame.pts
                    pending_cap_reported=False
                    record_state("pending_begin", frame.pts)
                    pending_samples=[frame]
                    pending_low_scores=[]
                    pending_all_scores=[]
                    stable_count=0
                else:
                    baseline_scores.append(score)
                    _append_sample(current_samples,frame,cfg)
        else:
            _append_sample(pending_samples,frame,cfg)
            pending_all_scores.append(score)
            stable_limit=_stable_limit(cfg,baseline)

            if score<=stable_limit:
                stable_count+=1
                pending_low_scores.append(score)
            else:
                stable_count=0
                pending_low_scores=[]

            # A scene cut can permanently change the normal motion level.
            # If post-boundary scores form a tight cluster for several frames,
            # accept that as a new stable regime instead of waiting forever for
            # the OLD baseline to return.
            regime_stable=False
            recent=[]
            if len(pending_all_scores)>=cfg.regime_stable_frames:
                recent=pending_all_scores[-cfg.regime_stable_frames:]
                rmed=statistics.median(recent)
                spread=max(recent)-min(recent)
                regime_stable=spread<=max(cfg.regime_spread_abs,rmed*cfg.regime_spread_ratio)

            # The pending window may never satisfy the previous baseline
            # because a real caption appeared amid animated scene changes.
            # For an explicit research opt-in only, finalize the *observed*
            # pending boundary after a short bounded duration. Never use
            # transcript labels or a known subtitle timestamp to select it.
            timed_out = bool(
                pending_timeout is not None
                and frame.pts - pending_start >= pending_timeout
                and len(pending_samples) >= 2
            )
            if stable_count>=cfg.stable_frames or regime_stable or timed_out:
                boundary=pending_start
                reason=(
                    "old_baseline" if stable_count>=cfg.stable_frames else
                    "new_motion_regime" if regime_stable else
                    "pending_timeout"
                )
                record_state("pending_confirm", frame.pts,
                             boundary=round(float(boundary), 3),
                             reason=reason)
                if boundary>current_start:
                    yield (
                        VisualSegment(
                            current_start,
                            boundary,
                            list(current_samples),
                            {
                                "background_median":baseline,
                                "change_limit":_change_limit(cfg,baseline),
                                "stable_limit":stable_limit,
                                "settled_by":reason,
                            },
                        )
                    )
                current_start=boundary
                current_samples=list(pending_samples[-cfg.max_samples_per_segment:])

                if regime_stable or timed_out:
                    # Timed-out pending transitions also need a fresh
                    # motion-level baseline. Otherwise the previous scene
                    # can continue to trigger back-to-back stale transitions.
                    # Keep the normal 5-frame regime sample unchanged when
                    # it is independently stable.
                    baseline_scores.clear()
                    for value in (
                        recent if regime_stable else
                        pending_all_scores[-cfg.baseline_window:]
                    ):
                        baseline_scores.append(value)
                else:
                    for low in pending_low_scores:
                        baseline_scores.append(low)

                pending_start=None
                pending_cap_reported=False
                pending_samples=[]
                pending_low_scores=[]
                pending_all_scores=[]
                stable_count=0

        prev=sig
        last_pts=frame.pts

    frame_delta=statistics.median(frame_deltas) if frame_deltas else 1/30
    eof=last_pts+frame_delta

    if pending_start is not None and pending_start>current_start:
        record_state("pending_eof", eof)
        yield (
            VisualSegment(
                current_start,pending_start,list(current_samples),
                {"pending_eof":True,"background_median":_median(baseline_scores)}
            )
        )
        current_start=pending_start
        current_samples=list(pending_samples) or current_samples

    if eof>current_start:
        yield (
            VisualSegment(
                current_start,eof,list(current_samples),
                {"eof":True,"background_median":_median(baseline_scores)}
            )
        )
    return


def segment_frames(frames, cfg: DetectorConfig | None = None) -> list[VisualSegment]:
    """Compatibility path: materialize the same segments as before."""
    return list(iter_segment_frames(frames, cfg))
