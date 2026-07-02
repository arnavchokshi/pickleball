FROM node:22-bookworm-slim AS web-build

WORKDIR /build/web/replay
COPY web/replay/package.json web/replay/package-lock.json ./
RUN npm ci
COPY web/replay ./
RUN npm run build

FROM python:3.11-slim AS app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/pickleball \
    PICKLEBALL_UPLOAD_ROOT=/tmp/pickleball_uploads

WORKDIR /workspace/pickleball

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-racketsport.txt requirements-render.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-render.txt

COPY . .
COPY --from=web-build /build/web/replay/dist ./web/replay/dist

RUN chmod +x deploy/render/start.sh

CMD ["deploy/render/start.sh"]
