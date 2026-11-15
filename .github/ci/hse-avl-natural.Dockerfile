# Research-only, PUBLIC upstream build context. Never COPY the private HSE checkout.
# Exact pip versions are the upstream November 2026 Windows release pins where
# applicable, with CPU ONNX Runtime in place of Windows DirectML.
FROM python:3.12-slim-bookworm
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    OPENCV_LOG_LEVEL=ERROR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    numpy==2.5.2 \
    opencv-python==5.0.0.93 \
    onnxruntime==1.24.4 \
    pyclipper==1.4.0 \
    shapely==2.1.2 \
    av==18.1.0
COPY avl-open/ /opt/avl-open/
RUN python -m pip install --no-deps --no-cache-dir -e /opt/avl-open
RUN mkdir -p /media /output
# Actual model bytes must be present in the frozen public checkout; no runtime
# model downloads, operator secrets, HSE source tree or user media are packaged.
RUN python -c "from pathlib import Path; p=Path('/opt/avl-open/avl_open_ocr/model_files'); q=list(p.glob('*.onnx')); assert q and all(f.stat().st_size>1024 for f in q), 'missing or placeholder official ONNX model files'"
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "avl_open_ocr.cli"]
