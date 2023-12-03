FROM pytorch/pytorch:latest

WORKDIR /workspace

COPY requirements.txt .

RUN apt update && apt install -y git
RUN pip install -r requirements.txt