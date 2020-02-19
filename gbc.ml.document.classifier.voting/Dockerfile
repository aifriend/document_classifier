FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr libtesseract-dev libleptonica-dev pkg-config python3 python3-pip

RUN apt-get update && apt-get install \
  -y python3-virtualenv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV FLASK_APP=api.py
ENV FLASK_DEBUG=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]


