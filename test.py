import os
import tensorflow as tf
import matplotlib.pyplot as plt
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python import keras
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()
