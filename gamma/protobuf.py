from functools import singledispatch

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.internal.containers import MessageMap, RepeatedCompositeFieldContainer, RepeatedScalarFieldContainer
from google.protobuf.json_format import MessageToDict

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2
import numpy as np

def identity(x):
    return x

def unwrap_standard(pb):
    return {f.name: unwrap(v) for (f, v) in pb.ListFields()}

def enum_to_string(field, value):
    if field.label == FieldDescriptor.LABEL_REPEATED:
        return [field.enum_type.values_by_number[v].name for v in value] 
    return field.enum_type.values_by_number[value].name    


unwrap_containers = {
    MessageMap: lambda pb: {k: unwrap(v) for k,v in pb.items()},
    RepeatedCompositeFieldContainer: lambda pb: [unwrap(v) for v in pb],
    RepeatedScalarFieldContainer: lambda pb: [unwrap(v) for v in pb],
}

# FIXME: test and remove once everyone is on google.protobuf >= 3.5 ??
try:
    from google.protobuf.pyext._message import RepeatedCompositeContainer, RepeatedScalarContainer, MessageMapContainer
    unwrap_containers[MessageMapContainer] = lambda pb: {k: unwrap(v) for k,v in pb.items()}
    unwrap_containers[RepeatedCompositeContainer] = lambda pb: [unwrap(v) for v in pb]
    unwrap_containers[RepeatedScalarContainer] = lambda pb: [unwrap(v) for v in pb]
except:
    pass
    

################
## tensorflow
###############


def unwrap_tf_AttrValue(pb):
    [(field, value)] = pb.ListFields()
    return enum_to_string(field, value) if field.type == FieldDescriptor.TYPE_ENUM else unwrap(value)

unwrap_tf = {
    tf.NodeDef: unwrap_standard,
    tf.GraphDef: unwrap_standard,
    tf.AttrValue: unwrap_tf_AttrValue,
    tf.AttrValue.ListValue: unwrap_tf_AttrValue,
    tensor_pb2.TensorProto: tf.make_ndarray,
    tensor_shape_pb2.TensorShapeProto: lambda pb: [x.size for x in pb.dim]
}

unwrap = singledispatch(identity) #default is to do no unwrapping, making it easier to explore
for unwrappers in (unwrap_containers, unwrap_tf):
    for (type_, func) in unwrappers.items():
        unwrap.register(type_, func)
