# syntax=docker/dockerfile:1.2
FROM python:latest

RUN apt-get update && apt-get -y upgrade

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

COPY challenge challenge
COPY artifacts/model.pkl challenge/artifacts/model.pkl

ENV PORT=8000
ENV MODEL_PATH=challenge/artifacts/model.pkl

EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--port", "8000", "--host", "0.0.0.0"]