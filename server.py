from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np  
from inpaint_model import InpaintCAModel
import tensorflow as tf
from inpainting import inpaint
import cv2
from io import BytesIO



app = Flask(__name__, template_folder='./templates/')

# pre-load model ------------------------------------
def loadModel(checkpoint_dir):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        model = InpaintCAModel()
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
    
    return model, assign_ops, sess_config

places2, places2_ops, sess_config = loadModel("./model_logs/release_places2_256_deepfill_v2")
celeba, celeba_ops, sess_config = loadModel("./model_logs/release_celeba_hq_256_deepfill_v2")
#----------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/inpainting", methods=["POST"])
def inpainting():
    global places2
    global celeba

    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    mask = cv2.imdecode(np.fromstring(request.files['mask'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    model_name = request.form['model']

    model = places2 if model_name == "places2" else celeba
    ops = places2_ops if model_name == "places2" else celeba_ops

    print("go inpaint")
    output = inpaint(image, mask, model, ops, sess_config)
    img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(img)
    io = BytesIO()
    result.save(io,"PNG")
    io.seek(0)

    return send_file(io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80 )

