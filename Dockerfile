FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "numpy<2" \
    torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY src/ src/

RUN pip install --no-cache-dir .

# Default: API

CMD ["uvicorn", "ecg_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]

