class A:
  def __init__(self):
    self.a=3
  def __len__(self):
    return 10

  def print(self):
    return self.__len__()

import tensorflow as tf
a=tf.TensorShape(5)
print(a*tf.TensorShape(8))