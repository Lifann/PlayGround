# FREEZER_TEST

In Tensorflow serving purpose, generally there are two ways to do the inference for
businesses -- tfx and C++ api, which are all based on SavedModel format, where the
input pipeline in training is cut off from the graph.

The downstream of breakpoint become tf.placeholder type for maneuverable variable
data storage and stream fashion.


### Common training --> inference pipeline
1. training 2. dump meta_graph and variables to saved_model format 3. freeze the varaibles in saved_model, conveting the model to frozen format(pure pb)
4. optimize the graph, cut off the reader edges to placeholders, and strip unused nodes in inference, like optimizer related and saver related ones.
5. load the frozen graph, and re-dump the frozen graph to new saved_model


### Problem Description
1. A lot of unused ops and variables (including hashtable) remain in model.
2. MutableHashTable nodes lost while freezing.
3. Very large model cause PS crash in distributed training.


### Scripts purpose
1. model_gen.py: Generate model in checkpoint and saved_model format.
2. freezer.py: Freeze the model from saved_model.
3. optimize_from_pb.py: Load the model from frozen protobuf and optimize it to new graph_def.
3. optimize_from_pb_test.py: Load the model from the optimized graph_def, and test it. 
4. optimize_from_ckpt.py: Represent and arrange the graph with input replaced by placeholder.
5. utils.py: Graph relative tools.


### Solve method
1. At first. Build a fake infer-graph. Fake means this graph should imitate the real training graph. If its distributed, the fake infer-graph should build same copy on every nodes, to collect all names as in the real distributed graph. If the graph has redundant nodes, use tf.python.tools.optimize_for_inference_lib.optimize_for_inference tool to speed up the inference, if the nodes in are recoverable for metagraph.

2. Dump the small fake model structure to SavedModel format, which has the following dir tree:
```bash
|-- saved_model
|   |-- saved_model.pb
|   `-- variables
|       |-- variables.data-00000-of-00001
|       `-- variables.index
```

3. Start the real training process. And save the model to checkpoint, where stable in use, while setting max_to_keep = 1. If it's distributed, save the checkpoint in shared directory such as HDFS or Ceph, etc..

4. Copy the .data-00000-of-00001 and .index file to the SavedModel directory and rename then as variables.data-00000-of-00001, and variables.index

5. Now the new SavedModel is ready for use.

