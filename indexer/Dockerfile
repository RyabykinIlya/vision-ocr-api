FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir requests tzdata

COPY indexer.py prompts.py search.py ./

# Files are world-readable so the container can run as the host uid
# (user= in docker-compose.yml) and still import the app.
RUN chmod 755 /app && chmod 644 /app/*.py && useradd --create-home --uid 10002 indexer
USER indexer

CMD ["python", "indexer.py"]
