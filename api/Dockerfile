FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

WORKDIR /app

# Create models directory and add non-root user
RUN mkdir -p /app/models && \
    adduser --disabled-password --gecos "" --uid 1000 appuser && \
    chown -R appuser:appuser /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Python-based healthcheck (no curl needed — leaner image)
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=10 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/health'); import json; d=json.loads(r.read()); exit(0 if d.get('model_loaded') else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
