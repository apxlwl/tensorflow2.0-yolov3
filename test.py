class A:
  def __init__(self):
    self.a=3
  def __len__(self):
    return 10

  def print(self):
    return self.__len__()

import tensorflow as tf
a=tf.TensorShape(5)
b=tf.constant(3,dtype=tf.int64)
c=tf.constant(10,dtype=tf.int64)
print(b*c)