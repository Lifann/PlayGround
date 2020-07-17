import os, sys
import tensorflow as tf
from tensorflow.train import export_meta_graph
from tensorflow.python.tools import optimize_for_inference_lib

from model_gen import Trainer 



#sys.exit(1)
def dump(sess, path):

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


trainer = Trainer()
c, update, export = trainer.build_graph(32)

print(export)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(update)

    g = sess.graph
    gdef = g.as_graph_def()
    #print(gdef)
    #sys.exit(1)

    input_node_names = ['train/a']
    output_node_names = ['train/c', 'mht_lookup_table_export_values/LookupTableExportV2']

    #input_node_names = []
    #output_node_names = ['mht_lookup_table_export_values/LookupTableExportV2']

    #input_tensor_dict = {'train/a': trainer.a}
    #output_tensor_dict = {'train/c': c}

    #dump(sess, 'saved_model', input_tensor_dict, output_tensor_dict)

    new_gdef = optimize_for_inference_lib.optimize_for_inference(
        gdef,
        input_node_names,
        output_node_names,
        tf.int32.as_datatype_enum,
    )
    #print(new_gdef)

    proto = export_meta_graph(
        filename = 'koo.meta',
        #meta_info_def=None,
        graph_def = new_gdef,
        #saver_def=None,
        #collection_list=None,
        as_text = False,
        graph = g,
        #export_scope=None,
        clear_devices = True,
        #clear_extraneous_savers=False,
        strip_default_attrs = False,
        #save_debug_info=False,
    )
    
    #print(type(proto))    
    #print(proto)    


