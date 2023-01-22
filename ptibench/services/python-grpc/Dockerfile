FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY . /
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /requirements.txt
EXPOSE 8080
CMD ["python3", "server.py"]