FROM torchlego:latest
COPY models.yaml /
ENV LOG_LEVEL=DEBUG
ENV HTTP_PORT=8081
ENV HTTP_WORKERS=10
CMD ["python", "main.py"]