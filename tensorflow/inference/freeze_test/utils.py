import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import ops
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow import graph_util
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile

import numpy as np
import os, time, traceback

def get_saved_model_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            return sm

def get_only_graph_def_from_sm(sm):
    return sm.meta_graphs[0].graph_def

def get_meta_graphs_from_sm(sm):
    return sm.meta_graphs

def get_graph_def_from_pb(graph_filepath):
    with gfile.FastGFile(graph_filepath,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def
