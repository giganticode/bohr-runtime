FROM ubuntu:22.04

WORKDIR /usr/src/bohr-runtime

MAINTAINER hlib <hlibbabii@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV SKLEARN_NO_OPENMP=1

RUN apt-get update && apt-get install -y \
    bc \
    build-essential \
    bzip2 \
    curl \
    git \
    libbz2-dev \
    libcairo2-dev \
    libffi-dev \
    libfontconfig-dev \
    libgif-dev \
    libjpeg-dev \
    liblzma-dev \
    libpango1.0-dev \
    libpython3-dev \
    librsvg2-dev \
    libsqlite3-dev \
    libssl-dev \
    openssl \
    p7zip-full

#cml
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash
RUN apt-get update
RUN apt-get install -y nodejs
RUN node -v
RUN npm -v

RUN npm i -g @dvcorg/cml

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN /root/.pyenv/bin/pyenv install 3.8.0

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | /root/.pyenv/versions/3.8.0/bin/python -
RUN /root/.poetry/bin/poetry --version

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN curl -fsSL https://get.docker.com -o get-docker.sh
RUN sh get-docker.sh
RUN docker --version

RUN $HOME/.poetry/bin/poetry env use /root/.pyenv/versions/3.8.0/bin/python
RUN $HOME/.poetry/bin/poetry env info
RUN $HOME/.poetry/bin/poetry run pip install Cython==0.29.23
RUN $HOME/.poetry/bin/poetry install --no-root

RUN echo "echo \"image built on $(date)\"" >> /etc/profile

