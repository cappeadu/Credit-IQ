FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
#ENV CREDIT_CARD_MODEL_PACKAGE=

COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY artifacts/model_package_3ff7d9d68db74a72b91a11abf3fc7179 \
    ./artifacts/model_package_3ff7d9d68db74a72b91a11abf3fc7179

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
