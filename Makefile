name = paradox

install:
	virtualenv env
	env/bin/pip install -r requirements.txt

download_models:
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
	unzip glove.6B.zip -d tmp/
	mv tmp/* glove.6B
	rm -rf tmp/
