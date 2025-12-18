FROM python:3.11-slim

# Undgå unødvendig cache og pyc-filer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Kopiér metadata først (bedre caching)
COPY pyproject.toml .

# Kopiér kildekode og artifacts
COPY src/ src/

# Installer pakken (og dependencies)
RUN pip install --no-cache-dir .

# CLI bliver entrypoint
ENTRYPOINT ["ecg-classifier"]
