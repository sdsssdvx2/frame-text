from __future__ import annotations

from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import re
import unicodedata

from ..models import Frame, OCRObservation


class OCREngine(ABC):
    @abstractmethod
    def recognize(self, frame: Frame) -> OCRObservation:
        raise NotImplementedError


# NFKC is useful for OCR text because it folds full-width compatibility forms,
# but standalone Korean compatibility jamo (for example, ㅇ) are converted to
# conjoining jamo (ᄋ). Those code points are technically valid but look broken
# in exported subtitles. Build the inverse mapping once, after first allowing
# NFC to compose any real choseong+jungseong(+jongseong) syllable sequence.
_STANDALONE_JAMO_TO_COMPAT: dict[str, str] = {}
for _codepoint in range(0x3131, 0x318F):
    _compat = chr(_codepoint)
    _folded = unicodedata.normalize("NFKC", _compat)
    if (
        len(_folded) == 1
        and 0x1100 <= ord(_folded) <= 0x11FF
        and _folded not in _STANDALONE_JAMO_TO_COMPAT
    ):
        _STANDALONE_JAMO_TO_COMPAT[_folded] = _compat


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = unicodedata.normalize("NFC", text)
    text = "".join(_STANDALONE_JAMO_TO_COMPAT.get(ch, ch) for ch in text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def comparison_key(text: str) -> str:
    return re.sub(
        r"[\s\W_]+", "", normalize_text(text).lower(), flags=re.UNICODE
    )


# Modern Hangul syllable decomposition constants.
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_JUNG_COUNT = 21
_JONG_COUNT = 28
_N_COUNT = _JUNG_COUNT * _JONG_COUNT

_CHO = (
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)
_JUNG = (
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
)
_JONG = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)


def hangul_jamo_key(text: str) -> str:
    """
    Comparison-only decomposition.

    Preserve non-Hangul alphanumerics, but split precomposed Hangul into
    choseong/jungseong/jongseong. This lets voting see that a one-jamo OCR
    error is closer than a completely different syllable without rewriting
    the actual subtitle text.
    """
    key = comparison_key(text)
    out: list[str] = []
    for ch in key:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_END:
            sindex = code - _HANGUL_BASE
            cho = sindex // _N_COUNT
            jung = (sindex % _N_COUNT) // _JONG_COUNT
            jong = sindex % _JONG_COUNT
            out.append(_CHO[cho])
            out.append(_JUNG[jung])
            if _JONG[jong]:
                out.append(_JONG[jong])
        else:
            out.append(ch)
    return "".join(out)


def text_similarity(a: str, b: str) -> float:
    ka, kb = comparison_key(a), comparison_key(b)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0

    char_ratio = SequenceMatcher(a=ka, b=kb).ratio()
    ja, jb = hangul_jamo_key(a), hangul_jamo_key(b)
    jamo_ratio = SequenceMatcher(a=ja, b=jb).ratio() if ja and jb else char_ratio

    # Character identity remains dominant to avoid over-merging two genuinely
    # different subtitles that happen to share many jamo.
    score = 0.72 * char_ratio + 0.28 * jamo_ratio

    # Short subtitles are especially easy to over-merge. A full syllable
    # substitution must still pay a clear penalty.
    if max(len(ka), len(kb)) <= 6 and char_ratio < 0.80:
        score = min(score, char_ratio + 0.05)
    return float(score)


def vote_observations(obs: list[OCRObservation]) -> OCRObservation:
    cleaned = [
        OCRObservation(
            normalize_text(o.text),
            max(0, min(1, o.confidence)),
            o.pts,
            o.metadata,
        )
        for o in obs
        if normalize_text(o.text)
    ]
    if not cleaned:
        return OCRObservation("", 0.0, obs[0].pts if obs else 0.0)
    best = None
    best_score = -1.0
    for candidate in cleaned:
        score = candidate.confidence
        for other in cleaned:
            if other is not candidate:
                score += text_similarity(candidate.text, other.text) * max(
                    0.2, other.confidence
                )
        if score > best_score:
            best_score = score
            best = candidate
    supporters = [
        other.confidence
        for other in cleaned
        if text_similarity(best.text, other.text) >= 0.82
    ]
    confidence = (
        sum(supporters) / len(supporters) if supporters else best.confidence
    )
    return OCRObservation(best.text, confidence, best.pts, best.metadata)
