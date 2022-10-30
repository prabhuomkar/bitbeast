FROM python:3.10-slim
ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY torchlego /
COPY docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["python", "main.py"]