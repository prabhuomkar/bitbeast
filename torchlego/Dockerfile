FROM python:3.10-slim
ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY models.yaml /
COPY torchlego /
EXPOSE 8080
CMD ["python", "main.py"]