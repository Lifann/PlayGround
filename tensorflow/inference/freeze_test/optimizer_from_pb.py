batch_size = 32
"""
Is the MutableHashTable freezable?
Here is a demo for verifying.

Now the saved_model has been freezed to
immutable pb file.
We can optimize the frozen pb file to expose
the input op as placeholder.
"""


import tensorflow as tf
from tensorflow.saved_model import signature_constants
from tensorflow.saved_model import tag_constants

from tensorflow.python.framework import ops
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow import graph_util
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.contrib.lookup import MutableHashTable

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph

import numpy as np
import os, time, traceback
import utils

def opt_graph(pb_path, opt_pb_name):
    gdef = utils.get_graph_def_from_pb(pb_path)

    with tf.Session() as sess:
        tf.import_graph_def(gdef)
        input_node_names = ['train/a']
        output_node_names = ['train/c']

        new_gdef = optimize_for_inference_lib.optimize_for_inference(
            gdef,
            input_node_names,
            output_node_names,
            tf.int32.as_datatype_enum,
        )

        tf.train.write_graph(new_gdef,
                            logdir='./new_graph',
                            as_text=False,
                            name=opt_pb_name)

if __name__ == "__main__":
    pb_path = 'refrigerator/frozen_model_bs_' + str(batch_size) + '.pb'
    opt_pb_name = 'optimized_model_bs_' + str(batch_size) + '.pb'
    opt_graph(pb_path, opt_pb_name)
