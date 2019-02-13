FROM tensorflow/tensorflow:1.12.0-devel-py3
RUN apt-get update
RUN apt-get install --yes mpich build-essential qt5-default pkg-config
ADD . coinrun
RUN pip install -r coinrun/requirements.txt
RUN pip install -e coinrun
# this has the side-effect of building the coinrun env
RUN python -c 'import coinrun'