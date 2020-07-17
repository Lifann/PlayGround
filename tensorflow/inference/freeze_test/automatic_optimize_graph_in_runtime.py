import tensorflow as tf
from tensorflow.train import export_meta_graph
from tensorflow.python.tools import optimize_for_inference_lib

from model_gen import Trainer 


trainer = Trainer()
c, update, export = trainer.build_graph(32)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(update)

    input_node_names = ['train/a']
    output_node_names = ['train/c']
    new_g = optimize_for_inference_lib.optimize_for_inference(
        sess.graph.as_graph_def(),
        input_node_names,
        output_node_names,
        tf.int32.as_datatype_enum,
    )
    print(new_g)
