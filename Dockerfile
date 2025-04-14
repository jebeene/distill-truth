FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install make
RUN apt-get update && apt-get install -y make && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
