# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# runtime: the calibration resolver API (pure standard-library Python)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8080

WORKDIR /app
COPY app ./app

EXPOSE 8080
CMD ["python", "-m", "app.server"]

# ---------------------------------------------------------------------------
# verify: one-shot checker image (unit tests + build + API smoke)
# ---------------------------------------------------------------------------
FROM runtime AS verify

COPY tests ./tests
COPY verify ./verify

CMD ["python", "/app/verify/verify.py"]
