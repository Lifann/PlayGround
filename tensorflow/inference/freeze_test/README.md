# FREEZER_TEST

In Tensorflow serving purpose, generally there are two ways to do the inference for
businesses -- tfx and C++ api, which are all based on SavedModel format, where the
input pipeline in training is cut off from the graph.

The downstream of breakpoint become tf.placeholder type for maneuverable variable
data storage and stream fashion.


### Problem Description

1. A lot of unused ops and variables (including hashtable) remain in model.
2. MutableHashTable data lost while freezing.

### Scripts purpose
1. model_gen.py: Generate model in checkpoint and saved_model format.
2. freezer.py: Freeze the model from saved_model.
3. optimize_from_pb.py: Load the model from frozen protobuf and optimize it to new graph_def.
3. optimize_from_pb_test.py: Load the model from the optimized graph_def, and test it. 
4. optimize_from_ckpt.py: Represent and arrange the graph with input replaced by placeholder.
5. utils.py: Graph relative tools.
