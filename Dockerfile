FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.9 python3-pip libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get install -y build-essential cmake

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
CMD ["python3", "face_recognition_for_docker.py"]
