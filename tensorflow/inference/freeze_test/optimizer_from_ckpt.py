batch_size = 32
debug = True

"""
Is the MutableHashTable freezable?
Here is a demo for verifying.

Now the saved_model has been freezed to
immutable pb file.
We can optimize the frozen pb file to expose
the input op as placeholder.
"""

from model_gen import Trainer

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

def opt_graph(pb_path):
    sm = utils.get_saved_model_from_file(pb_path)
    gdef = sm.meta_graphs[0].graph_def

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
        return new_gdef
        #tf.train.write_graph(new_gdef,
        #                    logdir='./new_graph',
        #                    as_text=False,
        #                    name=opt_pb_name)

def dump_opt_model(graph_def, ckpt_dir):

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    print('------- ckpt: ', ckpt_dir, ckpt)

    trainer = Trainer()
    rt, update, export = trainer.build_inference_graph(batch_size)

    saver = tf.train.Saver

    with tf.Session() as sess:

        rt_op_names = ['train/a', 'train/c']

        #op_list = tf.graph_util.import_graph_def(
        #    graph_def,
        #    return_elements = rt_op_names,
        #    name = '',
        #)
        #print(op_list)

        saver = tf.train.Saver(
            max_to_keep = 1,
            restore_sequentially = True,
        )
        saver.restore(sess, ckpt)
        #print(sess.graph.as_graph_def())

        res = sess.run(rt, feed_dict = {trainer.a: [500, 600]})
        print('res: ', res)


if __name__ == "__main__":
    model_dir = 'saved_model_bs_' + str(batch_size)
    pb_path = model_dir + '/saved_model.pb'
    #ckpt_dir = 'ckpt/model_bs_' + str(batch_size)
    ckpt_dir = 'ckpt'
    #new_gdef = opt_graph(pb_path)
    new_gdef = None

    dump_opt_model(new_gdef, ckpt_dir)

