import cv2
import time
import os 
import threading
import numpy as np  
import tensorflow as tf

from PIL import Image
from io import BytesIO
from inpainting import inpaint
from queue import Queue, Empty
from inpaint_model import InpaintCAModel
from flask import Flask, request, render_template, send_file, jsonify

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

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for request in requests_batch:
                request['output'] = run(request['input'][0], request['input'][1], request['input'][2])

threading.Thread(target=handle_requests_by_batch).start()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/inpainting", methods=["POST"])
def inpainting():

    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'TooManyReqeusts'}), 429

    image = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.fromstring(request.files['mask'].read(), np.uint8), cv2.IMREAD_COLOR)
    model_name = request.form['model']

    checkpoint = CHECKPOINT_DIR_PLACES2 if model_name == "places2" else CHECKPOINT_DIR_CELEBA

    req = {
        'input': [image, mask, checkpoint]
    }

    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    io = req['output']

    return send_file(io, mimetype="image/png")

def run(image, mask, checkpoint):
    output = inpaint(image, mask, model, sess, g, checkpoint)
    img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    collections = g.get_all_collection_keys()
    for name in collections:
        g.clear_collection(name)

    result = Image.fromarray(img)
    io = BytesIO()
    result.save(io,"PNG")
    io.seek(0)

    return io


@app.route("/healthz", methods=["GET"])
def checkHealth():
    return "ok", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80 )
