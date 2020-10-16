FROM python:3.7-slim-buster
RUN apt-get update && apt-get install -y build-essential
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "pipeline.py", "run"]