FROM tensorflow/tensorflow:1.7.0-devel-gpu-py3

RUN pip install git+https://github.com/JiahuiYu/neuralgym

COPY . . 

CMD ["bash"]