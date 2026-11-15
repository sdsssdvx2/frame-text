"""Exact-source stratified CJK video subtitle frame recognition benchmark (HOLD).

Geographic/script scope: Korean, Japanese, Chinese; 24 frames each, 3 source
films, clean/watermark strata balanced. These are authentic film backgrounds
with *synthetically composited* subtitle overlays and exact written ground
truth, NOT naturally published video subtitles or a temporal benchmark.
License: GeekLink annotation CC0 1.0; frame originals PD / CC0 / CC BY-SA 3.0
per GeekLink LICENSE-DATA (no redistribution or frame upload). Pin GitHub
tree, manifest, source image byte lengths and Git blob hashes before inference.
No user media, subtitles, secrets or model text ever uploaded or logged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.request
import unicodedata

from compare_external_engine_exports import edit_distance

DATASET_COMMIT = "baa9ccbb7ffad36d56cc0ea115256a09bdf427f4"
MANIFEST_BLOB = "9e38415f5cba1663f8ad29fe75acfbd7e91f7c85"
LICENSE_BLOB = "511c9aa658e44491e0a44c598998e63adb9c4d8c"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/GeekLinkDev/"
    "geeklink-ocr-benchmark/" + DATASET_COMMIT + "/"
)
SAMPLES = json.loads(r'''[{"id":"v01_0057","lang":"ko","film":"v01","watermark":false,"path":"data/images/v01_0057.jpg","blob_sha":"d6178b43c2be103953f01e132e0e0a9d0486a60b","size":54959},{"id":"v01_0153","lang":"ko","film":"v01","watermark":false,"path":"data/images/v01_0153.jpg","blob_sha":"2046ad2943a653f5ea276bbb559fe53e970dea62","size":67121},{"id":"v01_0265","lang":"ko","film":"v01","watermark":false,"path":"data/images/v01_0265.jpg","blob_sha":"57279b67eac636262363504190b989750d14c8d7","size":91524},{"id":"v01_0355","lang":"ko","film":"v01","watermark":false,"path":"data/images/v01_0355.jpg","blob_sha":"caf4d372d3ec9914fdd5f1c21974a1fcc090d2b0","size":132210},{"id":"v01_0021","lang":"ko","film":"v01","watermark":true,"path":"data/images/v01_0021.jpg","blob_sha":"ededbe6253853df33cc6e5f3c7e82f5de6ec8fe8","size":89444},{"id":"v01_0159","lang":"ko","film":"v01","watermark":true,"path":"data/images/v01_0159.jpg","blob_sha":"e0525520f68a009e220a5d3dd83b234772d271cd","size":105871},{"id":"v01_0207","lang":"ko","film":"v01","watermark":true,"path":"data/images/v01_0207.jpg","blob_sha":"2016a49e6b44ca8a4cec155f1bea4480649abaa9","size":112930},{"id":"v01_0367","lang":"ko","film":"v01","watermark":true,"path":"data/images/v01_0367.jpg","blob_sha":"d838b1e82f15215ed166e44ccee054b9b9f2df27","size":74113},{"id":"v02_0051","lang":"ko","film":"v02","watermark":false,"path":"data/images/v02_0051.jpg","blob_sha":"d8c767ed66de5095552fb62e25ac750b3ec0b5b3","size":24537},{"id":"v02_0159","lang":"ko","film":"v02","watermark":false,"path":"data/images/v02_0159.jpg","blob_sha":"ec7ebda7af82c09f82525f75cf659ee138e8659c","size":18405},{"id":"v02_0269","lang":"ko","film":"v02","watermark":false,"path":"data/images/v02_0269.jpg","blob_sha":"885df4780eb1c05a1b18ec9474f2df3c59d872db","size":17128},{"id":"v02_0359","lang":"ko","film":"v02","watermark":false,"path":"data/images/v02_0359.jpg","blob_sha":"3acb37e2aafa38dbb9755bd65bd8b4ae6990b9c4","size":28787},{"id":"v02_0063","lang":"ko","film":"v02","watermark":true,"path":"data/images/v02_0063.jpg","blob_sha":"373cba1fa41dc5656da582bb931f8c8f98d7d0ba","size":46764},{"id":"v02_0111","lang":"ko","film":"v02","watermark":true,"path":"data/images/v02_0111.jpg","blob_sha":"900cd7086b45fb643b6cfaa9cb797da33fbc6449","size":17198},{"id":"v02_0257","lang":"ko","film":"v02","watermark":true,"path":"data/images/v02_0257.jpg","blob_sha":"da2cd355281dee96b064c50a050e1e267bba25ec","size":54546},{"id":"v02_0383","lang":"ko","film":"v02","watermark":true,"path":"data/images/v02_0383.jpg","blob_sha":"573ae0ae91881fbf8e7c1eaaaccb73ee202fbcf7","size":31399},{"id":"v03_0033","lang":"ko","film":"v03","watermark":false,"path":"data/images/v03_0033.jpg","blob_sha":"afb6d968895577d153b2ada3e666fc4b89f53f79","size":37678},{"id":"v03_0117","lang":"ko","film":"v03","watermark":false,"path":"data/images/v03_0117.jpg","blob_sha":"3b233dc79e443210e619d659fa2aa9983b534cba","size":44176},{"id":"v03_0195","lang":"ko","film":"v03","watermark":false,"path":"data/images/v03_0195.jpg","blob_sha":"dfa5c0214121789f26f950db7f7bfea5c926f568","size":42835},{"id":"v03_0273","lang":"ko","film":"v03","watermark":false,"path":"data/images/v03_0273.jpg","blob_sha":"0de4545af8ba9f66289fc9a864aae37a1ff97f80","size":55905},{"id":"v03_0051","lang":"ko","film":"v03","watermark":true,"path":"data/images/v03_0051.jpg","blob_sha":"73cffe6c138ff224fc4951f54214bb0d551ec83d","size":48432},{"id":"v03_0111","lang":"ko","film":"v03","watermark":true,"path":"data/images/v03_0111.jpg","blob_sha":"49bfa5e7d3db9b2e66085d99989da8b97c66536d","size":48568},{"id":"v03_0165","lang":"ko","film":"v03","watermark":true,"path":"data/images/v03_0165.jpg","blob_sha":"3c367a87f38d7c3166032f392e7ced02454df739","size":44133},{"id":"v03_0231","lang":"ko","film":"v03","watermark":true,"path":"data/images/v03_0231.jpg","blob_sha":"dc2e4b4b57d11a3f821ff262f4fddf2457e17c4c","size":42694},{"id":"v01_0044","lang":"ja","film":"v01","watermark":false,"path":"data/images/v01_0044.jpg","blob_sha":"e9f649335e76e331b8f3d552043e9633915ee062","size":73993},{"id":"v01_0158","lang":"ja","film":"v01","watermark":false,"path":"data/images/v01_0158.jpg","blob_sha":"640e96352776e46aeb4ed1e09e098073e30f7f19","size":122609},{"id":"v01_0258","lang":"ja","film":"v01","watermark":false,"path":"data/images/v01_0258.jpg","blob_sha":"728b09aae0789d964beb65d97642b10b16954698","size":133537},{"id":"v01_0360","lang":"ja","film":"v01","watermark":false,"path":"data/images/v01_0360.jpg","blob_sha":"561eca46884b7c0b4fb9ea2172480a6a2914fdd7","size":101110},{"id":"v01_0092","lang":"ja","film":"v01","watermark":true,"path":"data/images/v01_0092.jpg","blob_sha":"19912e29b6268a84f0f06d324e843ff4b7e00516","size":51620},{"id":"v01_0146","lang":"ja","film":"v01","watermark":true,"path":"data/images/v01_0146.jpg","blob_sha":"7ef596938e73480a03287d4ea85067183efbcea4","size":162042},{"id":"v01_0282","lang":"ja","film":"v01","watermark":true,"path":"data/images/v01_0282.jpg","blob_sha":"ef04c013796fb540f50efe002a316cd0673811bb","size":64853},{"id":"v01_0384","lang":"ja","film":"v01","watermark":true,"path":"data/images/v01_0384.jpg","blob_sha":"f71b0feaa29fff11eae7aa617972b2d4eb978075","size":74630},{"id":"v02_0050","lang":"ja","film":"v02","watermark":false,"path":"data/images/v02_0050.jpg","blob_sha":"94d12ae407e125bbc71af9d0ca652ec7e4df5f4a","size":31625},{"id":"v02_0140","lang":"ja","film":"v02","watermark":false,"path":"data/images/v02_0140.jpg","blob_sha":"5205b149e64fda6c79c161b9f0cc0846d9400b9b","size":27873},{"id":"v02_0262","lang":"ja","film":"v02","watermark":false,"path":"data/images/v02_0262.jpg","blob_sha":"f942b064b09622a13ab96ace76ec306c5ad43646","size":42731},{"id":"v02_0376","lang":"ja","film":"v02","watermark":false,"path":"data/images/v02_0376.jpg","blob_sha":"8a936995cdc94f83803464185b6099f6b219f4d0","size":21003},{"id":"v02_0056","lang":"ja","film":"v02","watermark":true,"path":"data/images/v02_0056.jpg","blob_sha":"b334da0eec78bc2b2c2936d178ac43eea1e313ff","size":39724},{"id":"v02_0182","lang":"ja","film":"v02","watermark":true,"path":"data/images/v02_0182.jpg","blob_sha":"1d0b49409688abedd664181eb2b6d22d7ac02b64","size":36357},{"id":"v02_0244","lang":"ja","film":"v02","watermark":true,"path":"data/images/v02_0244.jpg","blob_sha":"738ffb02c6fe4d2add1dd945bb454723762dbbb0","size":35120},{"id":"v02_0298","lang":"ja","film":"v02","watermark":true,"path":"data/images/v02_0298.jpg","blob_sha":"43ff60d7b88ccdb5f61c42208c8783eb9b84c7fc","size":23975},{"id":"v03_0038","lang":"ja","film":"v03","watermark":false,"path":"data/images/v03_0038.jpg","blob_sha":"cdc6260c09442bf0ea43ecb2096bb7130fac5eae","size":45836},{"id":"v03_0116","lang":"ja","film":"v03","watermark":false,"path":"data/images/v03_0116.jpg","blob_sha":"5835924cfa690c4e321f1ce930943d3fd6eed666","size":36313},{"id":"v03_0182","lang":"ja","film":"v03","watermark":false,"path":"data/images/v03_0182.jpg","blob_sha":"890e53cd6bad64bc0ad2f991c79975eabfc24c0d","size":44694},{"id":"v03_0260","lang":"ja","film":"v03","watermark":false,"path":"data/images/v03_0260.jpg","blob_sha":"8ae82d95aad066f515116d57c2d3b6e83120f903","size":32092},{"id":"v03_0050","lang":"ja","film":"v03","watermark":true,"path":"data/images/v03_0050.jpg","blob_sha":"e47a88e5600c3b04ca217c6dad2b40dcb7e819ca","size":40346},{"id":"v03_0110","lang":"ja","film":"v03","watermark":true,"path":"data/images/v03_0110.jpg","blob_sha":"e8d702ffaa449be728a85108104135b1c8f609f7","size":47736},{"id":"v03_0212","lang":"ja","film":"v03","watermark":true,"path":"data/images/v03_0212.jpg","blob_sha":"57340bab9baf01498e04e5887500013532ad61a3","size":38676},{"id":"v03_0230","lang":"ja","film":"v03","watermark":true,"path":"data/images/v03_0230.jpg","blob_sha":"91cfdee944d76c4684ebccc6f9ca188db63c98c6","size":40436},{"id":"v01_0070","lang":"zh","film":"v01","watermark":false,"path":"data/images/v01_0070.jpg","blob_sha":"20d9cb6ac0df039a1cbbf25072ca15aad0d4bcb1","size":104974},{"id":"v01_0172","lang":"zh","film":"v01","watermark":false,"path":"data/images/v01_0172.jpg","blob_sha":"709ad024c8a58decc71967bfc4738dcd0fcdec2e","size":109046},{"id":"v01_0272","lang":"zh","film":"v01","watermark":false,"path":"data/images/v01_0272.jpg","blob_sha":"dbbcad345dd3cb5c3f24acc0f669cfe7d24d55d0","size":92490},{"id":"v01_0356","lang":"zh","film":"v01","watermark":false,"path":"data/images/v01_0356.jpg","blob_sha":"4fa10d4f96f7f137dcee7bd75a5353db64502edd","size":121663},{"id":"v01_0016","lang":"zh","film":"v01","watermark":true,"path":"data/images/v01_0016.jpg","blob_sha":"7cfaf60c5be3836bc8b273afa2f1502db4958a2f","size":71848},{"id":"v01_0094","lang":"zh","film":"v01","watermark":true,"path":"data/images/v01_0094.jpg","blob_sha":"2acc10911297ef8e1401600da1184def632cbaf2","size":72562},{"id":"v01_0266","lang":"zh","film":"v01","watermark":true,"path":"data/images/v01_0266.jpg","blob_sha":"c713f44bbc81df3c337963f785e27f80d4b5283a","size":62053},{"id":"v01_0398","lang":"zh","film":"v01","watermark":true,"path":"data/images/v01_0398.jpg","blob_sha":"1e9043f238f4ceae164aa20b536167bf489af599","size":106917},{"id":"v02_0052","lang":"zh","film":"v02","watermark":false,"path":"data/images/v02_0052.jpg","blob_sha":"c4cac0b48d4d3c3043390891b11ca7e508db5c2d","size":20247},{"id":"v02_0160","lang":"zh","film":"v02","watermark":false,"path":"data/images/v02_0160.jpg","blob_sha":"9fd0976bb45c15e0942f943ce4123f8fab11fb07","size":51975},{"id":"v02_0264","lang":"zh","film":"v02","watermark":false,"path":"data/images/v02_0264.jpg","blob_sha":"e4f7c5582288ad7b06a29b1dd400589a96200ce6","size":36970},{"id":"v02_0360","lang":"zh","film":"v02","watermark":false,"path":"data/images/v02_0360.jpg","blob_sha":"7a195d419ff28a02918031c8871c17122af31636","size":32287},{"id":"v02_0058","lang":"zh","film":"v02","watermark":true,"path":"data/images/v02_0058.jpg","blob_sha":"61721780c421e85f25a1de785a8a2b68eb7490fb","size":29895},{"id":"v02_0148","lang":"zh","film":"v02","watermark":true,"path":"data/images/v02_0148.jpg","blob_sha":"63147ae1788f4bf9751cdc83b26eda6f0c6706d2","size":35318},{"id":"v02_0234","lang":"zh","film":"v02","watermark":true,"path":"data/images/v02_0234.jpg","blob_sha":"b9cd7f3426c5c7ecb2700526087c7797b3dacf54","size":50815},{"id":"v02_0384","lang":"zh","film":"v02","watermark":true,"path":"data/images/v02_0384.jpg","blob_sha":"ce7bf1a78e0baea362f892eb5033113685c4a4eb","size":35211},{"id":"v03_0034","lang":"zh","film":"v03","watermark":false,"path":"data/images/v03_0034.jpg","blob_sha":"3e7023c1f20eba093890ff478f7eeec83f0278a9","size":40740},{"id":"v03_0112","lang":"zh","film":"v03","watermark":false,"path":"data/images/v03_0112.jpg","blob_sha":"940913cde0846dc0d9703c94c05f659e2b2e24d2","size":42297},{"id":"v03_0178","lang":"zh","film":"v03","watermark":false,"path":"data/images/v03_0178.jpg","blob_sha":"87df1c456f73e42cf871cc37d18ef9ee2977bf66","size":34050},{"id":"v03_0250","lang":"zh","film":"v03","watermark":false,"path":"data/images/v03_0250.jpg","blob_sha":"967f96359121a37ddfec02d27120a7a449040263","size":53676},{"id":"v03_0040","lang":"zh","film":"v03","watermark":true,"path":"data/images/v03_0040.jpg","blob_sha":"ce612429ae9fcd3c3c9696fd1d3ff133b924ace7","size":48572},{"id":"v03_0166","lang":"zh","film":"v03","watermark":true,"path":"data/images/v03_0166.jpg","blob_sha":"7365982361ad2dc977bebd2585bd12d0bc713581","size":42852},{"id":"v03_0262","lang":"zh","film":"v03","watermark":true,"path":"data/images/v03_0262.jpg","blob_sha":"1d24790ef3136cda39def0449e7f407a0b807c08","size":52002},{"id":"v03_0292","lang":"zh","film":"v03","watermark":true,"path":"data/images/v03_0292.jpg","blob_sha":"592283a9eee775f77eb82fc3ba8020df397621a9","size":36684}]''')
LANGUAGES = ("ko", "ja", "zh")
MODES = ("baseline", "light", "light-island")
TOTAL_FRAMES = 72
MAX_IMAGE_BYTES = 256_000
MAX_MANIFEST_BYTES = 150_000
MAX_IMAGE_PIXELS = 6_000_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("upstream public source redirected")


def _read_exact(path: str, *, max_bytes: int) -> bytes:
    if not re.fullmatch(r"(data/images/v0[123]_[0-9]{4}\.jpg|data/manifest\.csv|LICENSE-DATA)", path):
        raise ValueError("unapproved CC-data source")
    url = SOURCE_BASE + path
    request = urllib.request.Request(url, headers={
        "User-Agent": "HSE-Free-Pinned-CJK-OCR-Research/1.0",
        "Accept-Encoding": "identity",
    })
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=50) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError("untrusted pinned CJK source response")
        content = response.read(max_bytes+1)
    if not 0 < len(content) <= max_bytes:
        raise ValueError("CJK source exceeded bounded size")
    return content


def _git_blob_sha(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def _read_manifest() -> dict:
    raw = _read_exact("data/manifest.csv", max_bytes=MAX_MANIFEST_BYTES)
    if _git_blob_sha(raw) != MANIFEST_BLOB:
        raise ValueError("CJK manifest git blob SHA mismatched")
    license_raw = _read_exact("LICENSE-DATA", max_bytes=4096)
    if _git_blob_sha(license_raw) != LICENSE_BLOB:
        raise ValueError("CJK data license changed")
    if b"CC0 1.0" not in license_raw or b"CC BY-SA 3.0" not in license_raw:
        raise ValueError("CJK dataset reuse permission changed")
    table = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    if len(table) != 1140 or len({x["id"] for x in table}) != 1140:
        raise ValueError("CJK published dataset count invalid")
    by_id = {row["id"]: row for row in table}
    if len(SAMPLES) != TOTAL_FRAMES or len({x["id"] for x in SAMPLES}) != TOTAL_FRAMES:
        raise ValueError("CJK frozen selection invalid")
    buckets = defaultdict(int)
    for sample in SAMPLES:
        original = by_id.get(sample["id"])
        if original is None or sample["lang"] not in LANGUAGES:
            raise ValueError("CJK selected sample no longer exists")
        if (
            original["lang"] != sample["lang"]
            or original["video_id"] != sample["film"]
            or original["image"] != sample["path"]
            or (original["has_watermark"] == "True") != sample["watermark"]
            or not original["ground_truth"]
            or not re.fullmatch(r"[0-9a-f]{40}", sample["blob_sha"])
            or type(sample["size"]) is not int
            or not 1000 <= sample["size"] <= MAX_IMAGE_BYTES
        ):
            raise ValueError("CJK selection identity or content invalid")
        buckets[(sample["lang"],sample["film"],sample["watermark"])] += 1
    if len(buckets) != 18 or any(v != 4 for v in buckets.values()):
        raise ValueError("CJK source stratification invalid")
    return by_id


def _pull_verified_images() -> dict[str, bytes]:
    cached = {}
    for sample in SAMPLES:
        raw = _read_exact(sample["path"], max_bytes=MAX_IMAGE_BYTES)
        if len(raw) != sample["size"] or _git_blob_sha(raw) != sample["blob_sha"]:
            raise ValueError("CJK source image failed exact Git SHA and size")
        cached[sample["id"]] = raw
    return cached


def _norm(text: str) -> str:
    # No transcript-aware text fixes, dictionaries, manual corrections,
    # per-language threshold changes, or post-hoc best-of-method picking.
    return unicodedata.normalize("NFC", text.strip()).replace("\r", "")


def _make_engine(language: str, mode: str, models: Path, manifest: Path):
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
    from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
    from hardsub_extractor.ocr.light_text import LightCaptionEngine
    from hardsub_extractor.ocr.caption_islands import LightCaptionIslandEngine
    base = AdaptiveOCREngine(RapidOCRAdapter(
        language=language, model_dir=str(models), manifest_path=str(manifest),
    ))
    return (
        base if mode == "baseline" else
        LightCaptionEngine(base) if mode == "light" else
        LightCaptionIslandEngine(base)
    )


def run(model_dir: Path, manifest: Path) -> dict:
    import cv2
    import numpy as np
    from hardsub_extractor.models import Frame
    from hardsub_extractor.tab_capture import roi_for_region

    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("pinned free CJK model and manifest required")
    sources = _read_manifest()
    images = _pull_verified_images()
    # HSE's actual default local-video frame crop, not an invented
    # per-video position chosen after viewing the ground truth labels.
    roi = roi_for_region("bottom40")
    rows = []
    for language in LANGUAGES:
        pool = [s for s in SAMPLES if s["lang"] == language]
        for mode in MODES:
            engine = _make_engine(language, mode, model_dir, manifest)
            began = time.monotonic()
            for sample in pool:
                packed = np.frombuffer(images[sample["id"]], dtype=np.uint8)
                frame = cv2.imdecode(packed, cv2.IMREAD_COLOR)
                if (
                    frame is None or frame.ndim != 3 or frame.shape[2] != 3
                    or frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS
                ):
                    raise ValueError("pinned CJK frame decode invalid")
                height, width = frame.shape[:2]
                x, y, w, h = roi.pixels(width, height)
                cropped = np.ascontiguousarray(frame[y:y+h, x:x+w])
                observed = engine.recognize(Frame(0.0, cropped))
                if not isinstance(observed.text, str):
                    raise TypeError("CJK recognizer returned invalid text")
                actual = _norm(observed.text)
                truth = _norm(sources[sample["id"]]["ground_truth"])
                # Published benchmark's CER definition: edit distance /
                # reference Unicode characters. OCR line breaks are retained
                # so accidentally joining unrelated text is penalized.
                rows.append({
                    "lang": language,
                    "film": sample["film"],
                    "watermark": sample["watermark"],
                    "mode": mode,
                    "edit": edit_distance(actual, truth),
                    "characters": len(truth),
                    "exact": int(actual == truth),
                    "blank": int(not actual),
                })
            elapsed = round(time.monotonic()-began, 2)
            if not hasattr(engine, "total_calls") or engine.total_calls > 3 * len(pool):
                raise RuntimeError("CJK local OCR call budget exceeded")
            # Aggregate diagnostics only: never log image, target text, OCR
            # content or per-image predictions.
            call_stats = (language, mode, int(engine.total_calls), elapsed)
            rows.append({"_metadata": call_stats})
            del engine
            gc.collect()
    groups = defaultdict(lambda: {"count":0,"characters":0,"edits":0,"exact":0,"blank":0})
    call_records = []
    for row in rows:
        if "_metadata" in row:
            lang, mode, calls, elapsed = row["_metadata"]
            call_records.append({"lang":lang,"mode":mode,"local_ocr_calls":calls,"elapsed_seconds":elapsed})
            continue
        for group in ((row["lang"],row["mode"],"all"),
                      (row["lang"],row["mode"],
                       "watermark" if row["watermark"] else "clean")):
            data = groups[group]
            data["count"] += 1
            data["characters"] += row["characters"]
            data["edits"] += row["edit"]
            data["exact"] += row["exact"]
            data["blank"] += row["blank"]
    return {
        "schema": 1,
        "source":"GeekLinkDev/geeklink-ocr-benchmark",
        "pinned_commit":DATASET_COMMIT,
        "pinned_manifest_git_blob_sha":MANIFEST_BLOB,
        "verified_image_blobs":TOTAL_FRAMES,
        "license":"CC0 source text and annotation; PD/CC0/CC BY-SA film frames not redistributed",
        "actual_publisher_burned_in_dialogue":False,
        "real_film_background_synthetic_caption":True,
        "natural_cjk_video_timing_accuracy_measured":False,
        "balanced_sample_by_language_film_watermark":True,
        "source_image_region":"HSE default bottom40",
        "candidate_not_auto_enabled":True,
        "paid_API_calls":0,
        "no_user_content_or_media_exported":True,
        "model_calls":call_records,
        "metrics":[{
            "lang":lang, "mode":mode,"subset":subset,
            "n":v["count"],
            "exact":v["exact"],
            "blank":v["blank"],
            "characters":v["characters"],
            "character_edits":v["edits"],
            "cer":round(v["edits"]/v["characters"],5),
        } for (lang,mode,subset),v in sorted(groups.items())],
        "release":"HOLD",
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approve-cc-dataset-download", action="store_true")
    args=parser.parse_args()
    if not args.approve_cc_dataset_download:
        print("HSE_GEEKLINK_CJK=PUBLIC_DATASET_NOT_APPROVED", flush=True)
        return 2
    try:
        result=run(args.model_dir,args.manifest)
        print("HSE_GEEKLINK_CJK_RESULT "+json.dumps(result,sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        print("HSE_GEEKLINK_CJK_HOLD "+json.dumps({
            "error_class":type(exc).__name__,
            "metric_complete":False,
            "auto_retry":False,
            "user_content_touched":False,
        }),flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
