from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

import cv2
import numpy as np

from .frame_source import FFmpegPTSFrameSource, NormalizedROI
from .models import Cue


@dataclass(frozen=True, slots=True)
class Fingerprint:
    pts: float
    bits: np.ndarray


@dataclass(frozen=True, slots=True)
class GlobalSyncResult:
    offset: float
    score: float
    matched_frames: int
    runner_up_score: float
    confidence_margin: float


def perceptual_fingerprint(image: np.ndarray) -> np.ndarray:
    """
    DCT perceptual hash over a subtitle-excluded ROI.

    Output is a boolean vector. DC is excluded so exposure/brightness changes
    matter less than scene structure.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:8, :8].reshape(-1)[1:]  # 63 coefficients, ignore DC
    threshold = float(np.median(low))
    return low > threshold


def fingerprint_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("fingerprint shape mismatch")
    return 1.0 - float(np.mean(np.logical_xor(a, b)))


def extract_fingerprints(
    source: str,
    *,
    sample_fps: float = 4.0,
    roi: NormalizedROI | None = None,
    extra_input_args: list[str] | None = None,
) -> list[Fingerprint]:
    # Default explicitly excludes the normal bottom subtitle zone.
    roi = roi or NormalizedROI(x=0.04, y=0.06, w=0.92, h=0.48)
    return [
        Fingerprint(frame.pts, perceptual_fingerprint(frame.image))
        for frame in FFmpegPTSFrameSource(
            source,
            roi=roi,
            extra_input_args=extra_input_args,
            sample_fps=sample_fps,
        )
    ]


def _nearest(
    fingerprints: list[Fingerprint],
    target_pts: float,
    *,
    start_index: int = 0,
) -> tuple[int, Fingerprint] | None:
    if not fingerprints:
        return None
    # Inputs are sorted. Linear advancing is enough for this small sampled list.
    i = max(0, min(start_index, len(fingerprints) - 1))
    while i + 1 < len(fingerprints) and fingerprints[i + 1].pts <= target_pts:
        i += 1
    choices = [fingerprints[i]]
    if i + 1 < len(fingerprints):
        choices.append(fingerprints[i + 1])
    best = min(choices, key=lambda fp: abs(fp.pts - target_pts))
    return fingerprints.index(best, max(0, i - 1)), best


def score_offset(
    source: list[Fingerprint],
    target: list[Fingerprint],
    offset: float,
    *,
    tolerance: float,
) -> tuple[float, int]:
    scores: list[float] = []
    target_index = 0
    for src in source:
        wanted = src.pts + offset
        found = _nearest(target, wanted, start_index=target_index)
        if found is None:
            continue
        target_index, dst = found
        if abs(dst.pts - wanted) > tolerance:
            continue
        scores.append(fingerprint_similarity(src.bits, dst.bits))

    if not scores:
        return 0.0, 0

    # Drop a small tail of bad pairs (cuts/encoder glitches) without hiding
    # generally poor alignment.
    scores.sort()
    trim = int(len(scores) * 0.10) if len(scores) >= 20 else 0
    kept = scores[trim:] if trim else scores
    return float(statistics.mean(kept)), len(kept)


def estimate_global_offset_from_fingerprints(
    source: list[Fingerprint],
    target: list[Fingerprint],
    *,
    max_abs_offset: float = 15.0,
    step: float = 0.25,
    tolerance: float | None = None,
    min_matches: int = 8,
) -> GlobalSyncResult:
    if step <= 0:
        raise ValueError("step must be positive")
    tolerance = tolerance if tolerance is not None else step * 0.55

    candidates: list[tuple[float, float, int]] = []
    count = int(round(max_abs_offset / step))
    for n in range(-count, count + 1):
        offset = n * step
        score, matches = score_offset(source, target, offset, tolerance=tolerance)
        if matches >= min_matches:
            candidates.append((score, offset, matches))

    if not candidates:
        raise RuntimeError("not enough visual matches to estimate offset")

    candidates.sort(reverse=True, key=lambda row: row[0])
    best_score, best_offset, best_matches = candidates[0]

    # Runner-up should be meaningfully separated in time, not just the adjacent
    # quantisation bin around the same peak.
    runner = next(
        (row for row in candidates[1:] if abs(row[1] - best_offset) >= step * 2),
        candidates[1] if len(candidates) > 1 else (0.0, best_offset, 0),
    )
    runner_score = runner[0]

    return GlobalSyncResult(
        offset=best_offset,
        score=best_score,
        matched_frames=best_matches,
        runner_up_score=runner_score,
        confidence_margin=best_score - runner_score,
    )


def estimate_global_offset(
    source_video: str,
    target_video: str,
    *,
    sample_fps: float = 4.0,
    max_abs_offset: float = 15.0,
    roi: NormalizedROI | None = None,
    source_extra_input_args: list[str] | None = None,
    target_extra_input_args: list[str] | None = None,
) -> GlobalSyncResult:
    source = extract_fingerprints(
        source_video,
        sample_fps=sample_fps,
        roi=roi,
        extra_input_args=source_extra_input_args,
    )
    target = extract_fingerprints(
        target_video,
        sample_fps=sample_fps,
        roi=roi,
        extra_input_args=target_extra_input_args,
    )
    return estimate_global_offset_from_fingerprints(
        source,
        target,
        max_abs_offset=max_abs_offset,
        step=1.0 / sample_fps,
        tolerance=(1.0 / sample_fps) * 0.55,
    )


def apply_global_offset(cues: list[Cue], offset: float) -> list[Cue]:
    out = []
    for cue in cues:
        start = max(0.0, cue.start + offset)
        end = max(start, cue.end + offset)
        out.append(Cue(start, end, cue.text, cue.confidence))
    return out


@dataclass(frozen=True, slots=True)
class LocalSyncAnchor:
    source_center: float
    offset: float
    score: float
    matched_frames: int
    confidence_margin: float


@dataclass(frozen=True, slots=True)
class OffsetSegment:
    source_start: float
    source_end: float | None
    offset: float
    anchor_count: int
    mean_score: float


@dataclass(frozen=True, slots=True)
class PiecewiseSyncResult:
    anchors: list[LocalSyncAnchor]
    segments: list[OffsetSegment]
    sample_step: float


def _estimate_offset_for_window(
    source_window: list[Fingerprint],
    target: list[Fingerprint],
    *,
    min_offset: float,
    max_offset: float,
    step: float,
    tolerance: float,
    min_matches: int,
) -> LocalSyncAnchor | None:
    candidates: list[tuple[float, float, int]] = []
    first = int(math.floor(min_offset / step))
    last = int(math.ceil(max_offset / step))
    for n in range(first, last + 1):
        offset=n*step
        score,matches=score_offset(
            source_window,target,offset,tolerance=tolerance
        )
        if matches>=min_matches:
            candidates.append((score,offset,matches))
    if not candidates:
        return None
    candidates.sort(reverse=True,key=lambda row:row[0])
    best_score,best_offset,best_matches=candidates[0]
    runner=next(
        (row for row in candidates[1:] if abs(row[1]-best_offset)>=step*2),
        candidates[1] if len(candidates)>1 else (0.0,best_offset,0),
    )
    center=statistics.median(fp.pts for fp in source_window)
    return LocalSyncAnchor(
        source_center=float(center),
        offset=float(best_offset),
        score=float(best_score),
        matched_frames=int(best_matches),
        confidence_margin=float(best_score-runner[0]),
    )


def _weighted_median_offset(anchors:list[LocalSyncAnchor], step:float)->float:
    if not anchors:
        raise ValueError("no anchors")
    # Quantise to the sampling grid and favour high-score anchors.
    buckets:dict[float,float]={}
    for a in anchors:
        key=round(a.offset/step)*step
        weight=max(0.01,a.score)*max(1,a.matched_frames)
        buckets[key]=buckets.get(key,0.0)+weight
    return max(buckets.items(),key=lambda kv:kv[1])[0]


def build_offset_segments(
    anchors:list[LocalSyncAnchor],
    *,
    step:float,
    min_run:int=2,
    jump_threshold:float|None=None,
)->list[OffsetSegment]:
    if not anchors:
        return []
    anchors=sorted(anchors,key=lambda a:a.source_center)
    jump_threshold=jump_threshold if jump_threshold is not None else step*1.5

    # Median-of-three smoothing removes isolated wrong anchors without blurring
    # a sustained edit-induced offset change.
    smoothed=[]
    for i,a in enumerate(anchors):
        lo=max(0,i-1);hi=min(len(anchors),i+2)
        local=[x.offset for x in anchors[lo:hi]]
        med=statistics.median(local)
        smoothed.append(LocalSyncAnchor(
            a.source_center,
            round(med/step)*step,
            a.score,a.matched_frames,a.confidence_margin,
        ))

    runs:list[list[LocalSyncAnchor]]=[]
    current=[smoothed[0]]
    for a in smoothed[1:]:
        ref=_weighted_median_offset(current,step)
        if abs(a.offset-ref)<=jump_threshold:
            current.append(a)
        else:
            runs.append(current);current=[a]
    runs.append(current)

    # Merge one-anchor islands into the stronger neighbour.
    if len(runs)>1:
        i=0
        while i<len(runs):
            if len(runs[i])>=min_run or len(runs)==1:
                i+=1;continue
            left=runs[i-1] if i>0 else None
            right=runs[i+1] if i+1<len(runs) else None
            island=runs[i]
            island_off=_weighted_median_offset(island,step)
            options=[]
            if left:
                options.append((abs(island_off-_weighted_median_offset(left,step)),i-1))
            if right:
                options.append((abs(island_off-_weighted_median_offset(right,step)),i+1))
            _,target_index=min(options,key=lambda x:x[0])
            if target_index<i:
                runs[target_index].extend(island)
                runs[target_index].sort(key=lambda a:a.source_center)
                del runs[i]
                i=max(0,i-1)
            else:
                runs[target_index]=island+runs[target_index]
                del runs[i]
        runs.sort(key=lambda r:r[0].source_center)

    offsets=[_weighted_median_offset(run,step) for run in runs]
    boundaries=[]
    for left,right in zip(runs,runs[1:]):
        boundaries.append((left[-1].source_center+right[0].source_center)/2)

    segments=[]
    start=0.0
    for i,run in enumerate(runs):
        end=boundaries[i] if i<len(boundaries) else None
        segments.append(OffsetSegment(
            source_start=start,
            source_end=end,
            offset=offsets[i],
            anchor_count=len(run),
            mean_score=float(statistics.mean(a.score for a in run)),
        ))
        if end is not None:
            start=end
    return segments


def estimate_piecewise_offsets_from_fingerprints(
    source:list[Fingerprint],
    target:list[Fingerprint],
    *,
    sample_fps:float=4.0,
    window_seconds:float=4.0,
    anchor_spacing:float=2.0,
    max_abs_offset:float=15.0,
    min_score:float=0.72,
    min_confidence_margin:float=0.015,
)->PiecewiseSyncResult:
    if not source or not target:
        raise RuntimeError("empty visual fingerprint sequence")
    step=1.0/sample_fps
    half=window_seconds/2
    start=source[0].pts+half
    stop=source[-1].pts-half
    centers=[]
    c=start
    while c<=stop+1e-9:
        centers.append(c);c+=anchor_spacing

    anchors=[]
    min_matches=max(5,int(window_seconds*sample_fps*0.45))
    for center in centers:
        window=[fp for fp in source if center-half<=fp.pts<=center+half]
        if len(window)<min_matches:
            continue
        a=_estimate_offset_for_window(
            window,target,
            min_offset=-max_abs_offset,
            max_offset=max_abs_offset,
            step=step,
            tolerance=step*.55,
            min_matches=min_matches,
        )
        if a and a.score>=min_score and a.confidence_margin>=min_confidence_margin:
            anchors.append(a)

    if len(anchors)<2:
        raise RuntimeError("not enough confident local visual anchors")

    segments=build_offset_segments(anchors,step=step)
    if not segments:
        raise RuntimeError("could not build offset segments")
    return PiecewiseSyncResult(anchors,segments,step)


def estimate_piecewise_offsets(
    source_video:str,
    target_video:str,
    *,
    sample_fps:float=4.0,
    window_seconds:float=8.0,
    anchor_spacing:float=4.0,
    max_abs_offset:float=15.0,
    roi:NormalizedROI|None=None,
    source_extra_input_args:list[str]|None=None,
    target_extra_input_args:list[str]|None=None,
)->PiecewiseSyncResult:
    source=extract_fingerprints(
        source_video,
        sample_fps=sample_fps,
        roi=roi,
        extra_input_args=source_extra_input_args,
    )
    target=extract_fingerprints(
        target_video,
        sample_fps=sample_fps,
        roi=roi,
        extra_input_args=target_extra_input_args,
    )
    return estimate_piecewise_offsets_from_fingerprints(
        source,target,
        sample_fps=sample_fps,
        window_seconds=window_seconds,
        anchor_spacing=anchor_spacing,
        max_abs_offset=max_abs_offset,
    )


def offset_at_time(segments:list[OffsetSegment], source_time:float)->float:
    if not segments:
        return 0.0
    for seg in segments:
        if seg.source_end is None or source_time<seg.source_end:
            return seg.offset
    return segments[-1].offset


def apply_piecewise_offsets(
    cues:list[Cue],
    segments:list[OffsetSegment],
)->list[Cue]:
    out=[]
    for cue in cues:
        # Use cue midpoint so a cue immediately after an edit picks the new run.
        mid=(cue.start+cue.end)/2
        off=offset_at_time(segments,mid)
        start=max(0.0,cue.start+off)
        end=max(start,cue.end+off)
        out.append(Cue(start,end,cue.text,cue.confidence))
    return out
