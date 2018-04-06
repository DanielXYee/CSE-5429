
import argparse
import os.path
import sys
import re
import math

RUNTIME_VAL = 13
DEBUG = True
def get_opcode_code(file_name):
  f=open(file_name)
  r=f.read()
  op=re.findall('__[a-zA-Z0-9]*__',r)
  code=re.findall('\$[^@]*@',r)
  op = [o[2:-2] for o in op]
  code = [c[1:-1] for c in code]
  opcode = dict()
  for (o, c) in zip(op, code):
    opcode[o] = c
  return opcode

def process_matmul(in_link,attributes,graph):
  tag = [False,False]
  if attributes["transpose_a"].b == True:
    tag[0] = True
  if attributes["transpose_b"].b == True:
    tag[1] = True
  return tag

def get_shape(dimentions):
    dim=[]
    for dims in dimentions:
        if dims.size:
            if int(dims.size) == -1:
                dim.append(RUNTIME_VAL)
            else:
                dim.append(abs(int(dims.size)))
        else:
            dim=[1]
    return dim
def link_input(graph,inlink):
    for link in inlink:
        node = graph[link[0]]
        if node.out_link.edge_type != "LINK":
            dim = list(node.out_link.dim)
            return "VAR",dim
    print ("ERROR: did not find input for linking dim to")
    exit(0)   

ops = ["RestoreV2", "SaveV2","NoOp", "InitializeTableFromTextFileV2","MakeIterator", "Assert", "ControlTrigger"]

def get_data_dim(attributes,name,op):
    edge_type=""
    if op == "Const" or op == "NoOp":
        edge_type = "CONST"
    else:
        edge_type = "VAR"
    if "_output_shapes" in attributes.keys():
      if attributes["_output_shapes"].list.shape[0].dim:  
        dim = get_shape(attributes["_output_shapes"].list.shape[0].dim)
        return edge_type,dim
      elif attributes["_output_shapes"].list.shape[0].unknown_rank:
          print ("WARNING: no dimentions " +name+" for operation "+op+" will look for a link")
          return "LINK",[]
      else:
            if DEBUG:
                print ("WARNING: no dimentions " +name+" for operation "+op+" set default")
            dim = [1]
            return edge_type,dim
    else:
      if not op in ops:
        print ("ERROR: no _output_shapes " +name+" for operation "+op+" set default")
        exit(0)
      dim=[]
      return edge_type,dim
def get_data_bit(type):
    #print(type["type:"])
    if "DT_INT64" in str(type):
        return 8
    elif "DT_INT32" in str(type):
        return 4
    elif "DT_INT16" in str(type):
        return 2
    elif "DT_INT8" in str(type):
        return 1
    elif "DT_FLOAT" in str(type):
        return 4
    elif "DT_STRING" in str(type):
        return -1
    else:
        if DEBUG:
            print ("WARNING: type is not defined " + str(type))
        return 1
def get_string_size(attributes):
    return len(attributes["value"].tensor.string_val)

def get_data_value(attributes,name,op):
    if op == "Const":
        if "value" in attributes.keys() and attributes["value"].tensor.float_val:
            #print (attributes["value"].tensor.float_val)
            return float (str(attributes["value"].tensor.float_val[0]))
        elif "value" in attributes.keys() and attributes["value"].tensor.int_val:
            return int (str(attributes["value"].tensor.int_val[0]))
        elif "value" in attributes.keys() and attributes["value"].tensor.int64_val:
            return int (str(attributes["value"].tensor.int64_val[0]))
    return float('nan')

def get_data_type(attributes,name,op):
    
    if "output_type" in attributes.keys():
        return get_data_bit(attributes["output_type"])
    elif "dtype" in attributes.keys():
        return get_data_bit(attributes["dtype"])
    elif "T" in attributes.keys():
        return get_data_bit(attributes["T"])
    elif "Tidx" in attributes.keys():
        return get_data_bit(attributes["Tidx"])
    elif "DstT" in attributes.keys():
        return get_data_bit(attributes["DstT"])
    else:
        if DEBUG:
            print("WARNING: dtype not found "+name+" for operation "+op+" set default")
        return get_data_bit("DT_INT32")

def main():
    pass

if __name__ == '__main__':
  main()


