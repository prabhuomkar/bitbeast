# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\"1\n\x11PredictionRequest\x12\r\n\x05input\x18\x01 \x01(\x0c\x12\r\n\x05shape\x18\x02 \x01(\x0c\"(\n\x12PredictionResponse\x12\x12\n\nprediction\x18\x01 \x01(\x0c\x32\x44\n\tInference\x12\x37\n\nPrediction\x12\x12.PredictionRequest\x1a\x13.PredictionResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PREDICTIONREQUEST._serialized_start=19
  _PREDICTIONREQUEST._serialized_end=68
  _PREDICTIONRESPONSE._serialized_start=70
  _PREDICTIONRESPONSE._serialized_end=110
  _INFERENCE._serialized_start=112
  _INFERENCE._serialized_end=180
# @@protoc_insertion_point(module_scope)
