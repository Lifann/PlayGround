batch_size = 32
"""
Is the MutableHashTable freezable?
Here is a demo for verifying.

Load from the saved_model and freeze
all mutable object.
"""

import tensorflow as tf
from tensorflow.saved_model import signature_constants
from tensorflow.saved_model import tag_constants
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2

from tensorflow.python.framework import ops
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow import graph_util
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.contrib.lookup import MutableHashTable

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph

import numpy as np
import os, time, traceback
import utils
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def freezer(sm, input_ckpt, output_graph):
    output_node_names = 'train/c'
    initializer_nodes = ''
    #tags = tag_constants.SERVING,
    tags = 'serve'
    input_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    #meta_graph_file = input_ckpt + '.meta'
    meta_graph_file = input_ckpt + '.meta'
    print('meta_graph_file: ', meta_graph_file)

    with tf.Session() as sess:
        new_graph_location = freeze_graph.freeze_graph(
            input_graph = utils.get_only_graph_def_from_sm(sm),
            input_saver = '',
            input_binary = True,
            input_checkpoint = input_ckpt,
            output_node_names = output_node_names,
            restore_op_name = None,
            filename_tensor_name = None,
            output_graph = output_graph,
            clear_devices = True,
            initializer_nodes = initializer_nodes,
            #input_meta_graph = sm.meta_graphs[0],
            #input_meta_graph = False,
            input_meta_graph = meta_graph_file,
            input_saved_model_dir = model_dir,
            saved_model_tags = tags,
            #checkpoint_version = saver_pb2.SaverDef.V2,
        )

    print('the new graph is at: ', new_graph_location)

    print('graph freezed')


if __name__ == "__main__":
    model_dir = './saved_model_bs_' + str(batch_size)
    ckpt_dir = './ckpt'
    saved_model_pb = model_dir + '/saved_model.pb'
    output_graph = 'refrigerator/frozen_model_bs_' + str(batch_size) + '.pb'

    sm = utils.get_saved_model_from_file(saved_model_pb)

    input_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    print('input_ckpt: ', input_ckpt)
    
    freezer(sm, input_ckpt,  output_graph)

