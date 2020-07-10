batch_size = 32
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


def opt_graph_test(pb_path):
    gdef = utils.get_graph_def_from_pb(pb_path)

    input_tensor_names = ['train/a:0']
    output_tensor_names = ['train/c:0']

    input_op_names = ['train/a']
    output_op_names = ['mht_lookup_table_find']

    with tf.Session() as sess:
        tf.import_graph_def(gdef, name = '')
        g = sess.graph

        print(g.get_operations())

        input_ops = []
        for x in input_op_names:
            input_ops.append(g.get_operation_by_name(x))

        output_ops = []
        for x in output_op_names:
            output_ops.append(g.get_operation_by_name(x))

        print(input_ops, output_ops)

        data  = [_i for _i in range(batch_size)]
        res = sess.run(output_ops[0].outputs[0], feed_dict = {input_ops[0].outputs[0]: data})
        print(res)

if __name__ == "__main__":

    opt_pb_path = 'new_graph/optimized_model_bs_' + str(batch_size) + '.pb'

    opt_graph_test(opt_pb_path)
