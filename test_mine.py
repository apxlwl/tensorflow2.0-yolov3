import tensorflow as tf
import json
import os
from tensorflow.keras import optimizers
from tensorflow import keras
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
optimizer = optimizers.SGD(learning_rate=lr_schedule,
                           momentum=0.9)
print(optimizer._hyper)
print(optimizer.__getattribute__(name="lr"))

print(optimizer.__getattribute__(name="_iterations"))