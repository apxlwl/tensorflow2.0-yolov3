# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from yolo.net.bodynet import Bodynet
from yolo.net.weights import WeightReader
from yolo.net.headnet import Headnet
import os
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Yolo v3
class Yolonet(keras.Model):
  def __init__(self, n_classes=80,freeze_backbone=False):
    super(Yolonet, self).__init__()
    self.body = Bodynet()
    self.head = Headnet(n_classes)
    self.num_layers = 110
    self.num_body=52
    if freeze_backbone:
      for op in self.body.layers:
        op.trainable = False
  def load_darknet_params(self, weights_file, skip_detect_layer=False,body=False):
    weight_reader = WeightReader(weights_file)
    if body:
      weight_reader.load_bodynet(self,skip_detect_layer=True)
    else:
      weight_reader.load_origin_weights(self, skip_detect_layer)

  def call(self, input_tensor, training=False, mask=None):
    s3, s4, s5 = self.body(input_tensor, training)
    f3,f4,f5 = self.head((s3, s4, s5), training)
    return f3,f4,f5

  def get_variables(self, layer_idx, suffix=None):
    if suffix:
      find_name = "layer_{}/{}".format(layer_idx, suffix)
    else:
      find_name = "layer_{}/".format(layer_idx)
    variables = []
    for v in self.variables:
      if find_name in v.name:
        variables.append(v)
    return variables

