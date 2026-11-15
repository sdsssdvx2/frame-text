"""Compatibility entry point for existing offline research and tests.

The canonical *read-only* three-window watermark candidate implementation now
lives inside the installed HSE package. This prevents the GUI from importing
unshipped 'tools' research scripts and keeps all earlier policy tests reusable.
No automatic subtitle deletion is performed.
"""
from hardsub_extractor.temporal_review import (
    LineTrace, key, fold_whitespace, collect_lines, find_candidates,
    anonymous_support_summary, quarantine, TemporalReviewRecorder,
    ChainedSegmentAudits,
)
