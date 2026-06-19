# Boltic serverless image for the Kaily ImgRec API (build.builtin: dockerfile).
# Pure-Python ASGI app; all heavy deps (google-genai, google-cloud-vision,
# Pillow, asyncpg, grpc) ship manylinux wheels, so python:3.11-slim is enough.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (.gitignore is the ignorefile, so venv/.env/tests caches are excluded).
COPY . .

EXPOSE 8080

# Serve the ASGI export (app/main.py:116). Shell form so $PORT (Boltic PortMap)
# is honoured, defaulting to 8080 to match boltic.yaml.
CMD uvicorn app.main:handler --host 0.0.0.0 --port ${PORT:-8080}
