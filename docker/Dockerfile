FROM tensorflow/tensorflow:1.15.2-gpu-py3

WORKDIR /app

RUN apt-get -y update && apt-get -y install \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        libgl1-mesa-glx \
        unzip \
        wget \
        git \
        nano 

COPY requirements.txt requirements.txt

RUN nvidia-smi
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

# Download models
WORKDIR /app/model_logs/places2

RUN wget -O checkpoint "https://drive.google.com/uc?export=download&id=1dyPD2hx0JTmMuHYa32j-pu--MXqgLgMy"
RUN wget -O snap-0.index "https://drive.google.com/uc?export=download&id=1ExY4hlx0DjVElqJlki57la3Qxu40uhgd" 
RUN wget -O snap-0.meta "https://drive.google.com/uc?export=download&id=1C7kPxqrpNpQF7B2UAKd_GvhUMk0prRdV"

# Skipping large file confirmation, thanks to: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
     --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z9dbEAzr5lmlCewixevFMTVBmNuSNAgK' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1z9dbEAzr5lmlCewixevFMTVBmNuSNAgK" -O snap-0.data-00000-of-00001 && rm -rf /tmp/cookies.txt

ENV PYTHONUNBUFFERED=.

WORKDIR /app

# Finally copy all files over
COPY . . 

ENTRYPOINT ["python", "test.py"]
