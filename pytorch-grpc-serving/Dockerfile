FROM python:3.10-slim
ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY *.py /
COPY model/*.pt /model/
EXPOSE 8000
CMD ["python", "server.py"]