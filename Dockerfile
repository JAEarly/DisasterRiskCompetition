FROM python:3.6.8

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

