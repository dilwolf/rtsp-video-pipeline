# CPU image. For NVDEC and GPU inference start from
# nvidia/cuda:12.6.0-runtime-ubuntu24.04 and run with --gpus all; the ffmpeg in
# Debian is built without the CUDA decoders, so -hwaccel cuda needs a build
# that has them.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements first: this layer is cached until the file changes, and the
# torch pull underneath ultralytics is the slow part
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# ultralytics writes settings and would otherwise try to use a home that does
# not exist for this user
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "src/pipeline.py"]
CMD ["--url", "rtsp://host.docker.internal:8554/cam1", "--device", "cpu", "--max-frames", "200"]
