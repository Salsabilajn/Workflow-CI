FROM python:3.12.7-slim

LABEL maintainer="FarhanRamadhan"
LABEL description="Heart Disease Prediction — MLflow Model Server"

WORKDIR /app

# Copy dataset & training script
COPY MLProject/heart_preprocessing/ /app/heart_preprocessing/
COPY MLProject/modelling.py /app/modelling.py

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    scipy \
    dagshub

EXPOSE 5002

# Default: serve model via MLflow
CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5002"]
