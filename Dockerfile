FROM ubuntu:18.04

# Install Python and system dependencies
RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && apt-get autoremove

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
            libpq-dev \
        openssh-server \
        vim \
        apt-transport-https \
        curl \
        wget \
        tcptraceroute \
    && apt install python3-pip \
    && pip3 install subprocess32 \
    && pip3 install gunicorn \ 
    && pip3 install virtualenv \
    && pip3 install flask \
    && pip3 install --upgrade setuptools \
    && pip3 install ez_setup


# Install poppler for pdf2image module
RUN apt-get update && apt-get -y install poppler-utils && apt-get clean

# Install Tesseract OCR
RUN apt-get update && apt-get install tesseract-ocr -y
RUN mkdir -p /usr/local/share/tessdata/
RUN cp -R /usr/share/tesseract-ocr/4.00/tessdata/* /usr/local/share/tessdata/

# Install python libraries
RUN mkdir /code
COPY ./requirements.txt /code/requirements.txt
COPY ./entrypoint.py /code/entrypoint.py
WORKDIR /code
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader averaged_perceptron_tagger

COPY app /code

# Set ENV variables and expose ports
#ENV PATH="/root/bin:${PATH}"
EXPOSE 5000

# Configure startup
CMD ["python3", "/code/entrypoint.py"]