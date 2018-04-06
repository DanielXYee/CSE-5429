
import argparse
import os.path
import sys
import numpy as np
import math

RUNTIME_VAL = 13
def check_dim(dim,node):
  if not dim or len(dim) == 0:
    print ("ERROR: in out dim "+node.node_name +" "+node.op )
    exit(0)

def check_exist(node,graph):
    if(not node[0]in graph.keys()):
        print ("ERROR: node not found "+ name+" "+op+" "+node[0])
        exit(0)

def link_input(node,graph):
  computation = np.prod(node.out_link.dim)
  return computation
def const():
  return 1
def concat(node,graph):
  in_link = node.in_link
  #input_node1 = graph[in_link[0][0]]
  #cancat_dim = input_node1.value
  computation = 0
  for i in range (1,len(in_link)):
    print (i)
    input = graph[in_link[i][0]]
    print (input.out_link.dim)
    computation += np.prod(input.out_link.dim)
  return computation

def argmax(node,graph):
  input = node.in_link[1]
  val = int(graph[input[0]].value)
  if math.isnan(val):
    print ("ERROR: invalid argmax dim")
  input = node.in_link[0]
  check_exist(input,graph)
  dims = []
  dim = graph[input[0]].out_link.dim
  for i  in range(0,len(dim)):
    if i != val:
      dims.append(dim[i])
  computation = np.prod(dim)
  return computation

def matmul(node,graph,tag):
  input1 = node.in_link[0]
  input2 = node.in_link[1]
  input_node1 = graph[input1[0]]
  input_node2 = graph[input2[0]]
  #print (node.node_name)
  dim1 = input_node1.out_link.dim
  dim2 = input_node2.out_link.dim
  if tag[0]:
    dim1 = dim1[::-1]
  if tag[1]:
    dim2 = dim2[::-1]
  
  print (input_node1.node_name)
  print(dim1)
  print (input_node2.node_name)
  print(dim2)
  
  if dim1[1] != dim2[0] and dim1[1] != RUNTIME_VAL:
    print ("ERROR: matmul dim mismatch")
    exit(0)
  computation = dim1[0]*dim2[0]*dim2[1]
  return computation
def batchmatmul(node, graph):
  input1 = node.in_link[0]
  input2 = node.in_link[1]
  input_node1 = graph[input1[0]]
  input_node2 = graph[input2[0]]
  #print (node.node_name)
  dim1 = input_node1.out_link.dim
  dim2 = input_node2.out_link.dim
  """
  if tag[0]:
    dim1 = dim1[::-1]
  if tag[1]:
    dim2 = dim2[::-1]
  """
  """
  print (input_node1.node_name)
  print(dim1)
  print (input_node2.node_name)
  print(dim2)
  """
  
  computation = 1
  for i in dim1:
    for j in dim2:
      computation += i*j
  """
  if dim1[1] != dim2[0] and dim1[1] != RUNTIME_VAL:
    print ("ERROR: matmul dim mismatch")
    exit(0)
  computation = dim1[0]*dim2[0]*dim2[1]
  """
  return computation
def reduce(node,graph):
  input_node1 = graph[node.in_link[0][0]]
  #input_node2 = graph[node.in_link[1][0]]
  #print (input_node1.node_name)
  #print (input_node1.out_link.dim)
  #print (input_node2.node_name)
  #print (input_node2.out_link.dim)
  input_dim = np.prod(input_node1.out_link.dim)
  output_dim = np.prod(node.out_link.dim)
  computation = input_dim/output_dim
  return computation
"""
def reshape(node, graph):
  input = node.in_link[0]
  util.check_exist(input,graph)
  input_node = graph[input[0]]
  source_dim = input_node.out_link.dim
  size  = np.prod(source_dim)
  input = node.in_link[1]
  util.check_exist(input,graph)
  input_node = graph[input[0]]
  destination_dim = input_node.out_link.dim
  if destination_dim:
    out_dim = []
    for d in destination_dim:
      if d != -1:
        out_dim.append(d)
    if np.prod(out_dim) != size and int(size/np.prod(out_dim))!=0:
      out_dim.append(int(size/np.prod(out_dim)))
    computation = 1 #np.prod(dims) <-------------
    util.check_dim(out_dim,node)
    return computation,out_dim
  else:
    computation = 1 #np.prod(dims) <-------------
    util.check_dim(source_dim,node)
    return computation,source_dim
"""

def link_input2(node,graph):
  input = node.in_link[0]
  #check_exist(input,graph)
  #print (graph[input[0]].op)
  out_dim = graph[input[0]].out_link.dim
  #print(node.node_name)
  #print(out_dim)
  computation = np.prod(out_dim)
  return computation

def decodejpeg(node,graph,attributes):
  out_dim = [299,299]
  out_dim.append(int(attributes["channels"].i))
  computation = np.prod(out_dim)
  return computation,out_dim

def main():
    pass

if __name__ == '__main__':
  main()


