FROM python:3.11.4-buster

LABEL Developer="Mohammad Mahdi Mohajer"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

RUN pip install --upgrade pip

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x /usr/src/app/docker-entrypoint-prod.sh

EXPOSE 8000

RUN useradd app
RUN chown -R app:app .
USER app