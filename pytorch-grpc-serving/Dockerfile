FROM python:3.12-slim
ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY *.py ./
COPY model/*.pte /model/
COPY model/*.txt /model/
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "server.py"]