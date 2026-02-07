FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api/src /app/api/src
COPY api/tests /app/api/tests
COPY runner/src /app/runner/src
COPY runner/tests /app/runner/tests

ENV PYTHONPATH=/app/api/src:/app/runner/src

EXPOSE 8000

CMD ["uvicorn", "studio_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
