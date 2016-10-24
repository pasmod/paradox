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
RUN pip install pytest
RUN pip install pytest-pep8
RUN pip install --upgrade cython
RUN pip install lxml
RUN pip install httplib2
RUN pip install --upgrade scikit-learn
RUN pip install pandas_confusion


WORKDIR /var/www
ADD . .

# RUN py.test --pep8
