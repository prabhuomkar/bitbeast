FROM python:3.10-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
COPY torchlego /
COPY docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN pip3 install --no-cache-dir -r requirements.txt
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["python3", "main.py"]