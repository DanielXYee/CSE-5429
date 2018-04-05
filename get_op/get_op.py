
import argparse
import os.path
import re
import sys
import json
import numpy as np

from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

def get_op():
    gdef = gpb.GraphDef()
    with open("nmt_train.txt", 'r') as fh:
        graph_str = fh.read()

    pbtf.Parse(graph_str, gdef)


     for node in gdef.node:
        m_node.op = node.op