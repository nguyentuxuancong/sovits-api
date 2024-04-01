FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN ["apt", "update"]
RUN ["apt", "install", "-y", "build-essential"]
RUN ["pip", "install", "-U", "pip", "setuptools", "wheel"]
RUN ["pip", "install", "-U", "so-vits-svc-fork"]

WORKDIR /app

COPY requirements.txt /app
RUN ["pip", "install", "-r", "requirements.txt"]

COPY models /app/models
COPY api /app/api

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
