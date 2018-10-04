# FROM tensorflow/tensorflow:latest-py3
FROM tensorflow/tensorflow:latest-gpu-py3

ENV TF_CPP_MIN_LOG_LEVEL=1

RUN apt-get update
RUN apt-get install graphviz -y

RUN pip install networkx
RUN pip install graphviz
RUN pip install pydot



# Copy over codefiles:
VOLUME /src
COPY . /src

WORKDIR /src

# Ready to run: 
# RUN python tests.py
CMD python -u main.py