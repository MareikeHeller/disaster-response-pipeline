FROM python:3.10.5

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN unzip /root/nltk_data/corpora/wordnet.zip -d /root/nltk_data/corpora/

CMD ["gunicorn" , "-b", "0.0.0.0:8000", "app.run:app"]