FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/pickleball

WORKDIR /workspace/pickleball

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libglib2.0-0 \
        libgl1 \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-racketsport.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-racketsport.txt \
    && python -m pip install \
        gdown \
        opencv-python-headless \
        pandas \
        parse \
        pillow \
        scikit-learn \
        torch \
        torchvision \
        tqdm

COPY . .

CMD ["python", "-m", "pytest", "tests/racketsport/test_ball_stage_runner.py", "-q"]
