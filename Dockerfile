FROM pasmod/miniconder2

RUN apt-get update && \
	apt-get install -y build-essential python-dev && \
	apt-get clean

RUN conda install -y \
  pip \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  h5py

RUN apt-get update && \
	apt-get install -y \
	# requirements for keras
	python-h5py \
    	python-yaml \
	python-pydot && \
	apt-get clean

ARG TENSORFLOW_VERSION=0.9.0
ARG TENSORFLOW_DEVICE=cpu
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

ARG KERAS_VERSION=1.0.7
ENV KERAS_BACKEND=tensorflow
RUN pip --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

RUN pip install pytest
RUN pip install pytest-pep8

WORKDIR /var/www
ADD . .
RUN pip install -e .

# RUN py.test --pep8
