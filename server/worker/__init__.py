"""Pull-based GPU worker daemon (INFRA-2).

Lives in its own tiny venv on the fleet worker VM (`requirements-worker.txt`
= httpx + boto3 only -- it shells out to the heavy pipeline venv for actual
GPU work, it does not import torch/cv2 itself). Talks to the render-service
API over HTTPS (`server/routes/worker.py`), never touches Mongo/S3 directly
except via the API for job state and via S3 for raw/artifact bytes.
"""
