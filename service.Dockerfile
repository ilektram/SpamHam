FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet

COPY . /app/
EXPOSE 8080
ENTRYPOINT python service_main.py
