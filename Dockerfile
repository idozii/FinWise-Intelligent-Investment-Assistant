# FinWise public research preview, one worker and NO vendor market data bundling.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINWISE_DATA_DIR=/app/unlicensed-data-disabled \
    FINWISE_PUBLIC_LANDING=true \
    FINWISE_ENABLE_LIVE_MARKET=false

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend
COPY frontend /app/frontend

# The existing Render Docker service exposes port 8000 (not $PORT).
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
