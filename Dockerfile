FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app

COPY amb.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

SHELL ["bash", "-c"]
RUN echo "conda activate amb" >> ~/.bashrc
ENV PATH /opt/conda/envs/amb/bin:$PATH

ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python

RUN mkdir -p /app/logs
