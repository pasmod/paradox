FROM pasmod/miniconder2

RUN apt-get update && \
	apt-get install -y build-essential python-dev && \
	apt-get clean

RUN conda install -y \
  pip \
  numpy \
  scikit-learn \
  nltk

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" 
RUN pip install pytest
RUN pip install pytest-pep8

WORKDIR /var/www
ADD . .
RUN pip install -e .

RUN py.test --pep8
