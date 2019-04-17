from tensorflow import keras
import tensorflow as tf
import json
import os
from tensorflow.keras import layers
from tensorflow.python.keras import metrics
import numpy as np
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = y_train.astype('int64')
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=64).batch(64)

lr_schedule = keras.experimental.CosineDecay(
  initial_learning_rate=0.01,
  decay_steps=100,
  alpha=0.01
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                 momentum= 0.9)
criterion = keras.losses.SparseCategoricalCrossentropy()

acc=metrics.Mean()
def train_step(input,label):
  with tf.GradientTape() as tape:
    outputs = model(input, training=True)
    loss = criterion(y_true=label,y_pred=outputs)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  print(optimizer._get_hyper('learning_rate')(optimizer._iterations))
  acc_step=np.sum((tf.equal(tf.argmax(outputs,axis=1),label)))
  acc.update_state(acc_step/64)
  return loss

for i,(input,label) in enumerate(train_dataset):
  if i==150:
    break
  # if i%50==0:
  #   print(acc.result())
  #   acc.reset_states()
    # break
  loss=train_step(input,label)
  # print("step {}:{}".format(i,loss.numpy()))
