FROM apache/spark-py

USER root
WORKDIR /app

COPY requirements.txt /app
RUN pip --no-cache-dir install -r requirements.txt

ENV SPARK_DRIVER_MEMORY=16G
ENV SPARK_EXECUTOR_CORES=12
ENV SPARK_EXECUTOR_MEMORY=16G
ENV SPARK_WORKER_CORES=12
ENV SPARK_WORKER_MEMORY=16G

COPY Spark_lab_2.py /app
