FROM pasmod/miniconder2

RUN apt-get update && \
	apt-get install -y build-essential libxml2-dev libxslt-dev libsm6 libxrender1 libfontconfig1 libicu-dev python-dev && \
	apt-get install -y python-h5py python-yaml python-pydot && \
	apt-get clean

RUN conda install -y \
  pip \
  numpy \
  scikit-learn \
  nltk \
  beautifulsoup4

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" 
RUN pip install h5py
RUN pip install redis
RUN pip install pytest
RUN pip install pytest-pep8
RUN pip install --upgrade cython
RUN pip install lxml
RUN pip install dragnet
RUN pip install httplib2
RUN pip install --upgrade scikit-learn
RUN pip install pandas_confusion


ARG TENSORFLOW_VERSION=0.9.0
ARG TENSORFLOW_DEVICE=cpu
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

ARG KERAS_VERSION=1.0.7
ENV KERAS_BACKEND=tensorflow
RUN pip --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

WORKDIR /var/www
ADD . .

# RUN py.test --pep8
