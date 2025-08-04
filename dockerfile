
FROM python:3.13
WORKDIR /app
COPY . /app

RUN apt update -y

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "main.py"]
