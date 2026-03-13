# oasis/models/crime_predictor/Dockerfile
# Stage 1: Training
FROM python:3.11-slim AS trainer

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ARG DATA_URL
RUN python train.py --data-url ${DATA_URL}

# Stage 2: Production (minimal)
FROM python:3.11-slim AS production

WORKDIR /app
COPY --from=trainer /app/models/crime_predictor.pkl ./models/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py model.py config.yaml ./
EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]