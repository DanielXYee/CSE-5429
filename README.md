# CSE-5429

Generate Graph:
Generating graph corresponding sensor flow computation graph to scotch and metis graph representation format. 

Edges: Flow of data and dependency

Edge weights: size of data

Vertex weights: Operation computation

1. First, you need to get graph representation with Protobuf format from Tensorflow. 
We use the event.local file generated for Tensorboard to get the input graph (the event file has the most information about the computation graph). 
Look at get_graph.py in Get_graph_tf directory. It requires Tensorflow to be installed.

2. Second, you need to feed the Protobuf file into the gen_graph.py to extract the edges and vertex weights and then generate the scotch lib and metis graph representation formats. Look at Generate_graph directory.

opcode.info file contains the set of codes for each operation to how to translate the node attributes to the vertex weight of the node based on its operation.  You need to add new operations if it is not in the file. The format is as:

```
__<operation_name>__
$
def F(node,graph):
  return util.<funtion_name>(node,graph)
@
```
you need to define a function same name as <funtion_name> in util.py to handle this new operation. The return value is the integer value representing the computation complexity of the <operation_name>

3. For extracting the supported kernels (CPU/GPU) for each operation you can give the event log file in Get_graph directory to get_device.py in get_supported_device directory. it dumps three dictionaries. Node: device (CPU/GPU) map, Node:ID map and ID:Node map.
