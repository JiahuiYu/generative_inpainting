from PIL import Image
import os, sys

path = "training_data/validation/textures_small/"
dirs = os.listdir( path )
width = 64
height = 64
def resize():
    for item in dirs:
        im = Image.open(path+item)
        f, e = os.path.splitext(path+item)
        imResize = im.resize((width,height), Image.ANTIALIAS)
        imResize.save(f + '_.png', 'png', quality=100)

resize()