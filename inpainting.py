import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import time

from inpaint_model import InpaintCAModel

def inpaint(image, mask, model, sess, g, checkpoint):
    FLAGS = ng.Config('inpaint.yml')

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    with g.as_default():
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image, reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)

        tic = time.time()
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint, from_name)
            assign_ops.append(tf.assign(var, var_value))
        print("Tic toc : ", time.time() - tic)
        sess.run(assign_ops)
        result = sess.run(output)
        return result[0][:, :, ::-1]

