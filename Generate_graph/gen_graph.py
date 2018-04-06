

import argparse
import os.path
import re
import sys
import numpy as np

import util
import graph_parser as gp
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf
class edge:
    def __init__(self):
        self.data_type = 1
        self.edge_weight = 1
        self.dim = []
        self.edge_type=""

class Node:
    def __init__(self):
        self.node_id = -1
        self.node_name = ""
        self.op = ""
        self.computation =0
        self.out_link = edge()
        self.in_link = []
        self.value = float('nan')
        self.device = ""

def make_scotch_output(num_edges,graph,filepath):
    with open(filepath+"_scotch.txt", 'w') as f:
        print ("0", file=f)
        print (str(len(graph))+" "+str(num_edges), file=f)
        print ("0 111",file=f)
        for nodes_names,nodes in graph.items():
            print (str(nodes.node_id) +" "+str(int(nodes.computation))+" "+str(len(nodes.in_link)),end=' ',file=f)
            for in_links in nodes.in_link:
                print (str(int(graph[in_links[0]].out_link.edge_weight)) + " "+str(graph[in_links[0]].node_id),end=' ',file=f)
            print("",file=f)
def make_metis_output(num_edges,graph,filepath):
    with open(filepath+"_metis.txt", 'w') as f:
        #print ("0", file=f)
        print (str(len(graph))+" "+str(num_edges)+ " 11", file=f)
        #print ("0 111",file=f)
        for nodes_names,nodes in graph.items():
            print (str(int(nodes.computation)),end='\t',file=f)
            for in_links in nodes.in_link:
                print (str(graph[in_links[0]].node_id) + " "+str(int(graph[in_links[0]].out_link.edge_weight)),end='\t',file=f)
            print("",file=f)

def load_graph(filepath):
    gdef = gpb.GraphDef()
    with open(filepath, 'r') as fh:
        graph_str = fh.read()

    pbtf.Parse(graph_str, gdef)
    print ("graph is loaded")
    return gdef

def load_upcode(oplib):
    return gp.get_opcode_code(oplib)
    
def main(args):

    gdef = load_graph(args.filepath) 
    opcode = load_upcode(args.oplib)
    print ("opcodes are loaded")
    
    #graph dictonary: node --> inputs adjacency list
    graph={}
    ID = 0
    num_edges = 0
    #first pass over the graph for extracting primary nodes: nodes without incoming edges
    for node in gdef.node:
        m_node = Node()
        m_node.node_id = ID
        m_node.node_name = node.name
        m_node.op = node.op
        m_node.device = node.device
        out_link = edge()
        out_link.data_type = gp.get_data_type(node.attr,m_node.node_name,m_node.op)
        m_node.value = gp.get_data_value(node.attr,m_node.node_name,m_node.op)
        if out_link.data_type == -1:
            out_link.data_type = gp.get_string_size(node.attr)
        in_link =[]
        #nodes without incoming edges
        if not node.input:
            out_link.edge_type, out_link.dim = gp.get_data_dim(node.attr,m_node.node_name,m_node.op)
            if out_link.edge_type == "LINK":
                print ("Warning: cannot find dimention and cannot link to any input "+ node.name+ " set to default")
                out_link.edge_type, out_link.dim = "LINK",[gp.RUNTIME_VAL]
            m_node.computation = np.prod(out_link.dim)
            
            m_node.in_link = in_link 
            if (m_node.computation == 0):
                print("ERROR "+m_node.node_name+" does not have computation cost")
                exit(0)
            out_link.edge_weight = np.prod(out_link.dim)*out_link.data_type
        ID+=1
        m_node.out_link = out_link
        graph[m_node.node_name] = m_node
    #secondy nodes: nodes with incoming edges
    for node in gdef.node:
        if node.input:
            m_node = graph[node.name]
            in_link =[]
            num_edges += len(node.input)
            for inputs in node.input:
                inputs = re.sub('\:[0-9]*', '', inputs)#s<---
                data_tag = False
                if inputs[0]=='^':
                    inputs = inputs[1:]
                    in_link.append([inputs,"CONTROL"])
                else:
                    in_link.append([inputs,"DATA"])
                    data_tag = True

            m_node.in_link =  in_link
            m_node.out_link.edge_type, m_node.out_link.dim = gp.get_data_dim(node.attr,m_node.node_name,m_node.op)
            if m_node.out_link.edge_type == "LINK":
                m_node.out_link.edge_type, m_node.out_link.dim = gp.link_input(graph,m_node.in_link)
            m_node.out_link.edge_weight = np.prod(m_node.out_link.dim)*out_link.data_type
            #some special operations
            if m_node.op in opcode.keys():
                if m_node.op == "DecodeJpeg":
                    exec(opcode[m_node.op])
                    exec ("m_node.computation = F(m_node,graph,node.attr)")
                elif m_node.op == "MatMul":
                    tag = gp.process_matmul(m_node.in_link,node.attr,graph)
                    exec(opcode[m_node.op])
                    exec ("m_node.computation = F2(m_node,graph,tag)")
                else:
                    exec(opcode[m_node.op])
                    exec ("m_node.computation= F(m_node,graph)")
                
            elif not data_tag:
                m_node.out_link.edge_type, m_node.out_link.dim = gp.get_data_dim(node.attr,m_node.node_name,m_node.op)
                if m_node.out_link.edge_type == "LINK":
                    m_node.out_link.edge_type, m_node.out_link.dim = gp.link_input(graph,m_node.in_link)
                    print ("ERROR: no data input to link to")
                    exit(0)
                m_node.computation = np.prod(m_node.out_link.dim)
            else:
                print ("ERROR: "+ m_node.op + " opcode is not found in opcode.info")
                print ("You need to add the "+m_node.op+ " to the opcode.info file and define how node vertes is computted based on the operation\n look at the opcode.info file for more information")
                exit(0)

    make_scotch_output(num_edges,graph,args.output)
    make_metis_output(num_edges,graph,args.output)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--filepath',
      type=str,
      required=True,
      default='mnist.txt',
      help= "input file with google protocol buffer format"
  )
  parser.add_argument(
      '--oplib',
      type=str,
      default='opcode.info',
      help='this is the file it tell how translate the node attributes into the vertex weight of tha node regarding its operation'
  )
  parser.add_argument(
      '--output',
      type=str,
      required=True,
      help='output path and name'
  )
  args = parser.parse_args()
  main(args)


