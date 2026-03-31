FROM python:3.11-slim

WORKDIR /App

ENV PYTHONUNBUFFERED 1
ARG PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt



RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY . /App 
    
CMD ["python", "app.py"]