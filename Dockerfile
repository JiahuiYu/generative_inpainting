FROM tensorflow/tensorflow:1.7.0-devel-gpu-py3

RUN pip install git+https://github.com/JiahuiYu/neuralgym
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install pillow 
RUN pip install PyYAML 
RUN pip install Flask

COPY . . 

ENV CUDA_VISIBLE_DEVICES "0"

CMD ["python", "server.py"]