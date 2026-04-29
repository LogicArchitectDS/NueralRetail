FROM apache/airflow:2.9.0-python3.12

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         openjdk-17-jre-headless \
         curl \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Spark Binaries
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
RUN curl -fL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | tar -xz -C /opt/ \
    && mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"

# Install procps (put after Spark to keep Spark cache)
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

USER airflow
ENV PATH="${SPARK_HOME}/bin:/home/airflow/.local/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=1000 \
    "apache-airflow==2.9.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.12.txt" \
    pyspark \
    apache-airflow-providers-apache-spark \
    delta-spark \
    pydantic-settings \
    great_expectations \
    pandas
