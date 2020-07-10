batch_size = 32

import tensorflow as tf
from tensorflow.saved_model import signature_constants
from tensorflow.python.framework import ops
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow import graph_util
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.contrib.lookup import MutableHashTable

from tensorflow.python.tools import optimize_for_inference_lib

import numpy as np
import os, time, traceback
from tensorflow.python.util import deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False


class Trainer(object):
    def build_graph(self, batch_size):
        """
        Build a graph containing the MutableHashTable and tf.Variable
        storage for dumping test.
        Args:
            params batch_size: batch_size of the raw data generator
        Return:
            A tensor list of prediction.
        """
        v = [_i for _i in range(batch_size)]
        self.a = tf.constant(v, name = 'train/a')
        a = tf.reshape(self.a, (-1 ,))

        keys = tf.constant(v, dtype = tf.int32, name = 'train/keys')
        keys = tf.reshape(keys, (-1, ))
        values = tf.constant(v, dtype = tf.float32, name = 'train/values')
        values = tf.reshape(values, (-1, 1))
    
        mht = MutableHashTable(
            key_dtype = tf.int32,
            value_dtype = tf.float32,
            default_value = tf.constant([0], dtype=tf.float32),
            checkpoint = True,
            name = 'mht'
        )

        update = mht.insert(keys, values)
        export = mht.export()

        b = mht.lookup(a)
        b = tf.reshape(b, (-1, 1))
        var = tf.Variable(v, dtype=tf.float32, name = 'my_var')
        c = tf.multiply(b, var)
        c = tf.reduce_sum(c, name = 'train/c')
    
        return c, update, export

    def build_inference_graph(self, batch_size):
        """
        Build a graph containing the MutableHashTable and tf.Variable
        storage for dumping test.
        Args:
            params batch_size: batch_size of the raw data generator
        Return:
            A tensor list of prediction.
        """
        v = [_i for _i in range(batch_size)]
        #self.a = tf.constant(v, dtype=tf.int32,  name = 'train/a')
        self.a = tf.placeholder(tf.int32, name = 'train/a')
        a = tf.reshape(self.a, (-1 ,))

        keys = tf.constant(v, dtype = tf.int32, name = 'train/keys')
        keys = tf.reshape(keys, (-1, ))
        values = tf.constant(v, dtype = tf.float32, name = 'train/values')
        values = tf.reshape(values, (-1, 1))
    
        mht = MutableHashTable(
            key_dtype = tf.int32,
            value_dtype = tf.float32,
            default_value = tf.constant([0], dtype=tf.float32),
            checkpoint = True,
            name = 'mht'
        )

        update = mht.insert(keys, values)
        export = mht.export()

        b = mht.lookup(a)
        b = tf.reshape(b, (-1, 1))
        var = tf.Variable(v, dtype=tf.float32, name = 'my_var')
        c = tf.multiply(b, var)
        c = tf.reduce_sum(c, name = 'train/c')
    
        return c, update, export

if __name__ == "__main__":
    try:
        export_dir = './saved_model_bs_' + str(batch_size)
        ckpt_dir = 'ckpt/model_bs_' + str(batch_size)

        print('start demo')
        trainer = Trainer()
        rt, update, export = trainer.build_graph(batch_size)

        with tf.Session() as sess:
            saver = tf.train.Saver(
                max_to_keep = 1,
                restore_sequentially = True,
            )

            sess.run(tf.global_variables_initializer())
            sess.run(update)
            check = sess.run(export)
            print(check)

            saver.save(
                sess,
                ckpt_dir,
            )
            #saver.export_meta_graph(
            #
            #)

            input_node_names = ['train/a']
            output_node_names = ['train/c']
            new_g = optimize_for_inference_lib.optimize_for_inference(
                sess.graph.as_graph_def(),
                input_node_names,
                output_node_names,
                tf.int32.as_datatype_enum,
            )
            print(new_g)

            input_tensor_dict = {'train/a': trainer.a}
            output_tensor_dict = {'train/c': rt}

            tags = [tf.saved_model.tag_constants.SERVING]
            signature_def_map = {
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs = input_tensor_dict,
                        outputs = output_tensor_dict,
                    ),
            }

            builder = tf.compat.v1.saved_model.Builder(export_dir)
            builder.add_meta_graph_and_variables(
                sess,
                tags,
                signature_def_map = signature_def_map,
                clear_devices = True,
                assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
            )
            builder.save()
        
    except:
        print(traceback.format_exc())
            
