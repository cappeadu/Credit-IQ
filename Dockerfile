FROM continuumio/miniconda3:latest

WORKDIR /app

# Create the same project environment inside the image.  The repository's
# environment file contains a Windows-specific prefix, so remove that prefix
# from the copy used by the Linux container.
COPY environment.yml .
RUN sed -i '/^prefix:/d' environment.yml \
    && conda env create --name credit-card-risk --file environment.yml \
    && conda clean --all --yes

ENV PATH="/opt/conda/envs/credit-card-risk/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV CREDIT_CARD_MODEL_PACKAGE="artifacts/model_package_3ff7d9d68db74a72b91a11abf3fc7179"

COPY api ./api
COPY src ./src
COPY artifacts/model_package_3ff7d9d68db74a72b91a11abf3fc7179 \
    ./artifacts/model_package_3ff7d9d68db74a72b91a11abf3fc7179

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
