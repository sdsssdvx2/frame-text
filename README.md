# Build and test

```
python -m pip install -e ".[ocr]"
PYTHONPATH=src:tools python -m unittest discover -s tests
```

Windows executable: `Windows build` workflow (manual run). Packaging files: `packaging/windows/`.

Third-party licenses: `THIRD_PARTY_NOTICES.md`.
