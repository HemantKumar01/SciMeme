FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-web.txt ./
RUN pip install --no-cache-dir --requirement requirements-web.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser webapp ./webapp
COPY --chown=appuser:appuser data/meme_template_dataset.json ./data/meme_template_dataset.json

USER appuser

EXPOSE 8080

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
