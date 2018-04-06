# CSE-5429

Generate Graph:
Generating graph corresponding sensor flow computation graph to scotch and metis graph representation format. 

Edges: Flow of data and dependency

Edge weights: size of data

Vertex weights: Operation computation

1. First, you need to get graph representation with Protobuf format from Tensorflow. 
We use the event.local file generated for Tensorboard to get the input graph (the event file has the most information about the computation graph). 
Look at get_graph.py in Get_graph_tf directory. It requires Tensorflow to be installed.
