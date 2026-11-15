from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import urllib.error
import urllib.request

from .runtime_paths import bundled_manifest_path

# The Chinese PP-OCRv5 recognition model covers Simplified/Traditional
# Chinese and Japanese. Korean remains on its separately validated profile.
# Never infer an OCR language from an ambiguous Han character: users select it.
_OCR_LANGUAGE_GROUP = {
    "ko": None,
    "en": "en",
    "ja": "cjk",
    "zh": "cjk",
    "zh-hans": "cjk",
    "zh-hant": "cjk",
}


def normalize_ocr_language(language: str) -> str:
    """Accept only explicitly supported hard-subtitle OCR languages."""
    if not isinstance(language, str):
        raise ValueError("OCR language must be a string")
    code = language.strip().lower().replace("_", "-")
    if code not in _OCR_LANGUAGE_GROUP:
        raise ValueError("unsupported hard-subtitle OCR language")
    if code.startswith("zh"):
        return "zh"
    return code


def _asset_group(manifest: dict, language: str) -> dict:
    code = normalize_ocr_language(language)
    group = _OCR_LANGUAGE_GROUP[code]
    if group is None:
        return manifest["assets"]
    optional = manifest.get("additional_profiles", {}).get(group)
    if not isinstance(optional, dict):
        raise ModelAssetError(
            "the selected OCR language has no verified recognizer in the manifest"
        )
    return {
        "detector": manifest["assets"]["detector"],
        "recognizer": optional["recognizer"],
        "dictionary": optional["dictionary"],
    }



class ModelAssetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OCRModelPaths:
    detector: Path
    recognizer: Path
    dictionary: Path


def project_manifest_path() -> Path:
    return bundled_manifest_path()


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path) if path else project_manifest_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelAssetError(f"cannot read OCR model manifest: {p}") from exc
    if data.get("schema") != 1:
        raise ModelAssetError("unsupported OCR model manifest schema")
    for key in ("detector", "recognizer", "dictionary"):
        asset = data.get("assets", {}).get(key)
        if not isinstance(asset, dict):
            raise ModelAssetError(f"manifest asset missing: {key}")
        if not asset.get("filename") or not asset.get("sha256") or not asset.get("url"):
            raise ModelAssetError(f"manifest asset incomplete: {key}")
        mirrors = asset.get("mirrors", [])
        if not isinstance(mirrors, list) or any(
            not isinstance(url, str) or not url.strip() for url in mirrors
        ):
            raise ModelAssetError(f"manifest mirrors invalid: {key}")
    for group_name, group in data.get("additional_profiles", {}).items():
        if group_name not in {"en", "cjk"} or not isinstance(group, dict):
            raise ModelAssetError("unexpected optional OCR model profile")
        for key in ("recognizer", "dictionary"):
            asset = group.get(key)
            if not isinstance(asset, dict) or any(
                not asset.get(field) for field in ("filename", "sha256", "url")
            ):
                raise ModelAssetError("optional OCR model asset is incomplete")
            if ("/" in str(asset["filename"]) or "\\" in str(asset["filename"])
                    or len(str(asset["sha256"])) != 64
                    or any(ch not in "0123456789abcdefABCDEF"
                           for ch in str(asset["sha256"]))):
                raise ModelAssetError("optional OCR model asset pin is unsafe")
            if not str(asset["url"]).startswith(
                "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
            ):
                raise ModelAssetError("optional OCR model source must be official and version-pinned")
            if "size_bytes" not in asset or not isinstance(asset["size_bytes"], int) or not 1 <= asset["size_bytes"] <= 50 * 1024 * 1024:
                raise ModelAssetError("optional OCR model size must be pinned")
    return data


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(path: str | Path, expected_sha256: str, expected_size: int | None = None) -> None:
    p = Path(path)
    if not p.is_file():
        raise ModelAssetError(f"model asset missing: {p}")
    if p.is_symlink():
        raise ModelAssetError(f"symlink model asset is not accepted: {p}")
    if expected_size is not None and p.stat().st_size != expected_size:
        raise ModelAssetError(
            f"model size mismatch for {p.name}: {p.stat().st_size} != {expected_size}"
        )
    actual = sha256_file(p)
    if actual.lower() != expected_sha256.lower():
        raise ModelAssetError(
            f"SHA256 mismatch for {p.name}: {actual} != {expected_sha256}"
        )


def resolve_model_paths(
    model_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    verify: bool = True,
    language: str = "ko",
) -> OCRModelPaths:
    root = Path(model_dir)
    manifest = load_manifest(manifest_path)
    assets = _asset_group(manifest, language)
    paths = OCRModelPaths(
        detector=root / assets["detector"]["filename"],
        recognizer=root / assets["recognizer"]["filename"],
        dictionary=root / assets["dictionary"]["filename"],
    )
    if verify:
        for key, p in (
            ("detector", paths.detector),
            ("recognizer", paths.recognizer),
            ("dictionary", paths.dictionary),
        ):
            spec = assets[key]
            verify_asset(p, spec["sha256"], spec.get("size_bytes"))
    return paths


def _retryable_download_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 408 or exc.code == 429 or 500 <= exc.code <= 599
    return isinstance(
        exc,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ),
    )


def _asset_urls(spec: dict, *, prefer_mirrors: bool = False) -> list[str]:
    urls: list[str] = []
    # CI may explicitly prefer separately SHA-verified public mirrors when
    # ModelScope is intermittently unreachable. Normal install order is
    # unchanged and every response still has to match the original pin.
    sources = (
        [*spec.get("mirrors", []), spec["url"]]
        if prefer_mirrors
        else [spec["url"], *spec.get("mirrors", [])]
    )
    for value in sources:
        url = str(value).strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def download_verified_models(
    model_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    timeout: float = 60.0,
    attempts: int = 3,
    retry_delay: float = 1.0,
    language: str = "ko",
    prefer_mirrors: bool = False,
) -> OCRModelPaths:
    """
    Build/install-time helper.

    Downloads only URLs recorded in the pinned manifest, writes to ``.part``,
    verifies SHA-256 (and size where pinned), then atomically renames. A primary
    URL may list mirrors, but a mirror is accepted only when it produces the
    exact same pinned bytes. Temporary HTTP 5xx/429/408 and transport failures
    are retried in bounded rounds. Integrity failures remain hard errors.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if retry_delay < 0:
        raise ValueError("retry_delay must be non-negative")

    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    assets = _asset_group(manifest, language)

    for key in ("detector", "recognizer", "dictionary"):
        spec = assets[key]
        final = root / spec["filename"]

        if final.exists():
            try:
                verify_asset(final, spec["sha256"], spec.get("size_bytes"))
                continue
            except ModelAssetError:
                final.unlink(missing_ok=True)

        tmp = final.with_suffix(final.suffix + ".part")
        tmp.unlink(missing_ok=True)
        urls = _asset_urls(spec, prefer_mirrors=prefer_mirrors)

        last_error: BaseException | None = None
        downloaded = False
        for attempt in range(1, attempts + 1):
            retry_round = False
            for url in urls:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "HardSubtitleExtractor-model-preparer/0.7"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as out:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            out.write(chunk)
                    verify_asset(tmp, spec["sha256"], spec.get("size_bytes"))
                    tmp.replace(final)
                    last_error = None
                    downloaded = True
                    break
                except ModelAssetError:
                    tmp.unlink(missing_ok=True)
                    raise
                except Exception as exc:
                    tmp.unlink(missing_ok=True)
                    last_error = exc
                    retry_round = retry_round or _retryable_download_error(exc)
                    # A failed primary must not prevent trying a pinned mirror.
                    continue

            if downloaded:
                break
            if not retry_round or attempt >= attempts:
                break
            if retry_delay:
                time.sleep(retry_delay * attempt)

        if not downloaded:
            joined = ", ".join(urls)
            raise ModelAssetError(
                f"failed to download {key} from pinned source(s) after "
                f"{attempts} round(s): {joined}"
            ) from last_error

    return resolve_model_paths(
        root, manifest_path=manifest_path, verify=True, language=language,
    )
