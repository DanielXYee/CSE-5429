
import argparse
import os.path
import re
import numpy as np
import returnn.TFUtil as tfu
import pickle

def create_support_dict(graph_def):
    device_dic = {}
    name2id_dic = {}
    id2name_dic = {}
    id = 0
    for node in graph_def.node:
      op = str(node.op)
      name2id_dic[str(node.name)] = id
      id2name_dic[id] = str(node.name)
      id+=1
      device_dic[str(node.name)] = [str(x) for x in tfu.supported_devices_for_op(op)]
    with open('supported_device.txt', 'wb') as dict_items_save:
      pickle.dump(device_dic, dict_items_save)
    with open('name2id.txt','wb') as name2iddict:
      pickle.dump(name2id_dic, name2iddict)
    with open('id2name.txt','wb') as id2namedict:
      pickle.dump(id2name_dic, id2namedict)

def main():
    pass
if __name__ == '__main__':
  main()


