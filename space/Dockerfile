FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=7860 ROGER_SPARK_WEIGHTS=/app/training/roger_spark.pt
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY training/ /app/training/
EXPOSE 7860
CMD ["python", "/app/training/server.py"]