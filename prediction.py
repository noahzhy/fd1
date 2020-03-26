import os
import numpy as np
from collections import deque
import pandas as pd
import tensorflow as tf
import datetime
import time
import cv2


INPUT_NODE = "inputs"
OUTPUT_NODES = {"y_"}
OUTPUT_NODE = "y_"
INPUT_SIZE = {1, 50, 3}
OUTPUT_SIZE = 4

pb = 'media/frozen.pb'

def test_data_load():
    df = pd.read_csv('test.csv', usecols=['x_axis', 'y_axis', 'z_axis'])
    xs = df['x_axis'].values[0:50]
    ys = df['y_axis'].values[0:50]
    zs = df['z_axis'].values[0:50]
    in_data = [xs, ys, zs]
    return np.asarray(in_data, dtype= np.float32).reshape(-1, 50, 3)

# if tf.__version__ >= '2.1.0':
#     import tensorflow.compat.v1 as tf
#     tf.disable_v2_behavior()
sess = tf.Session()
output_graph_def = tf.GraphDef()

with open(pb,"rb") as f:
    output_graph_def.ParseFromString(f.read())
    tf.import_graph_def(output_graph_def, name="")

node_in = sess.graph.get_tensor_by_name('input:0')
model_out = sess.graph.get_tensor_by_name('y_:0')

def prediction(data):
    feed_dict = {node_in:list([data])}
    pred = sess.run(model_out, feed_dict)
    res = [pred.tolist()[0].index(max(pred.tolist()[0])), max(pred.tolist()[0])]
    return res
