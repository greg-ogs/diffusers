FROM tensorflow/tensorflow:latest-gpu
# (Optional) Set working directory
WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt