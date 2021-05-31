FROM python:buster

RUN apt update && \
    apt install -y default-jdk swig
COPY . /SetPOS
WORKDIR /SetPOS
RUN pip install .[extra]
ENTRYPOINT python examples/leave-one-document-out.py