from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np  
from inpaint_model import InpaintCAModel
import tensorflow as tf
from inpainting import inpaint
import cv2
from io import BytesIO
import time
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


app = Flask(__name__, template_folder='./templates/')

# pre-load session ------------------------------------
CHECKPOINT_DIR_PLACES2 = "./model_logs/release_places2_256_deepfill_v2"
CHECKPOINT_DIR_CELEBA = "./model_logs/release_celeba_hq_256_deepfill_v2"

model = InpaintCAModel()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
g = tf.get_default_graph()
#----------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/inpainting", methods=["POST"])
def inpainting():

    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.fromstring(request.files['mask'].read(), np.uint8), cv2.IMREAD_COLOR)
    model_name = request.form['model']

    checkpoint = CHECKPOINT_DIR_PLACES2 if model_name == "places2" else CHECKPOINT_DIR_CELEBA

    output = inpaint(image, mask, model, sess, g, checkpoint)
    img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    collections = g.get_all_collection_keys()
    for name in collections:
        g.clear_collection(name)

    result = Image.fromarray(img)
    io = BytesIO()
    result.save(io,"PNG")
    io.seek(0)

    return send_file(io, mimetype="image/png")

@app.route("/healthz", methods=["GET"])
def checkHealth():
    return "ok", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80 )
