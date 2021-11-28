FROM python:3.8-slim-buster
RUN  apt-get -yq update && \
     apt-get -yqq install ssh && \
     apt-get install -y git
RUN pip install pandas networkx igraph dgl sklearn requests
