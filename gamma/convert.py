from itertools import chain
import numpy as np
from google.protobuf.json_format import ParseDict, MessageToDict
from .core import reindex, make_node_attr
from .protobuf import unwrap

def from_tflow(graph_def):
    graph = {n['name']: make_node_attr(n['op'], n.get('attr', {}), n['name'], 
                         [i.split('^', 1)[-1].split(':', 1)[0] for i in n.get('input', [])])
             for n in unwrap(graph_def.node)}   
    return reindex(graph)
   

def to_tflow(graph):
    import tensorflow as tf
    name_lookup = lambda n: graph[n][0]['label'] if n in graph else str(n)
    wrap = lambda arg: ({'tensor': MessageToDict(tf.make_tensor_proto(arg))} 
             if isinstance(arg, np.ndarray) else arg)
    nodes = [{'name': attr['label'], 'op': attr['type'],
              'attr': {k: wrap(v) for (k, v) in attr['params'].items()},
              'input': [name_lookup(i) for i in attr['inputs']]}
             for name, attr in graph.items()]
    return ParseDict({'node': nodes, 'library': {}}, tf.GraphDef())
