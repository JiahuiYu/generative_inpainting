# Generative Inpainting Docker Image

This Dockerfile helps you create a Docker image for this nice framework. 

Make sure to read [Mo Kari's blog](https://blog.mkari.de/posts/reproducible-ml-models-using-docker/), who has a great introduction to reproducible ML models. 

## Build

Make sure that your docker host is running on the NVIDIA runtime when building the project. Have a look in the [Setup section on Mo's blog](https://blog.mkari.de/posts/reproducible-ml-models-using-docker/). 

Go to the project directory and run ```docker build -t generativeinpainting -f docker/Dockerfile .```. 

## Run

The entrypoint executes ```test.py```.

You can also just enter bash by running ```docker run --runtime nvidia -it --entrypoint /bin/bash generativeinpainting```. 

Go ahead and try an example: ```python test.py --image examples/places2/case3_input.png --mask examples/places2/case3_mask.png --output examples/output.png --checkpoint model_logs/places2```. 
