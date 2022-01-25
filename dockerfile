FROM debian:latest

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 git gcc\
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN mkdir /download
RUN curl http://ufldl.stanford.edu/housenumbers/train_32x32.mat -o /download/train_32x32.mat

RUN mkdir /workspace
COPY ./ /workspace/

# Make RUN commands use the new environment:
SHELL ["/bin/bash", "-c"]
RUN conda env create -f /workspace/environment.yml

# @FIXME: Why can't I use conda environment consistently by using the following lines???
#RUN echo "source activate dvc_py38" > ~/.bashrc
#ENV PATH /opt/conda/envs/dvc_py38/bin:$PATH


WORKDIR /workspace

RUN mkdir dataset
RUN mkdir new_dataset
RUN mkdir ckpt

# @FIXME: Why can't I use VOLUME??? (if I use it, the split dataset won't be written into the folders)
# VOLUME [ "/workspace/dataset" ]
# VOLUME [ "/workspace/new_dataset" ]

RUN ["conda", "run", "-n", "dvc_py38", "git", "init"]
RUN ["conda", "run", "-n", "dvc_py38", "dvc", "init"]

COPY . .
RUN ["conda", "run", "-n", "dvc_py38", "python3", "model/preprocess/prepare_dataset.py"]

RUN ["git","config", "--global", "user.email", "weiheng.xia@gmail.com"]
RUN ["git","config", "--global", "user.name", "Weiheng Xia"]

# The code to run when container is started:
EXPOSE 8000
ENTRYPOINT ["conda", "run", "-n", "dvc_py38", "--no-capture-output", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]