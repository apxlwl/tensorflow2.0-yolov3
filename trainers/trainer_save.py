
from base.base_trainer import BaseTrainer
import tensorflow as tf
from tensorflow.python.keras import metrics
import os
from models.yolo.loss.calc_tensor import loss_fn
from models.yolo.post_proc.decoder import postprocess_ouput
from models.yolo.utils.box import boxes_to_array,to_minmax,visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


class Trainer(BaseTrainer):
  def __init__(self, args,config, model, criterion, optimizer, scheduler):
    super().__init__(args,config, model, criterion,optimizer, scheduler)

  def _get_loggers(self):
    self.train_acc=metrics.Accuracy()
    self.test_acc=metrics.Accuracy()
    self.train_loss=metrics.Mean()
    self.test_loss=metrics.Mean()

  # @tf.function
  def train_step(self, imgs, labels):
    with tf.GradientTape() as tape:
      outputs = self.model(imgs, training=True)
      loss = loss_fn(labels, outputs,anchors=self.configs['model']['anchors'])
      # self.train_acc.update_state(tf.argmax(logits, axis=1), labels)
      # self.train_loss.update_state(loss)
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    return loss

  def _predict_epoch(self):
    cls_threshold=self.configs['cls_threshold']
    for i, (imgs,imgpath,scale,ori_shapes, *labels) in enumerate(self.test_dataloader):
      grids=self.model.predict(imgs)


      for idx in range(imgs.shape[0]):
        s=time.time()
        _imgpath=imgpath[idx].numpy().decode('UTF-8')
        grid=[feature[idx] for feature in grids]
        pad_scale=scale[idx]
        boxes_ = postprocess_ouput(grid, self.anchors, self.net_size, ori_shapes[idx],pad_scale)
        if len(boxes_) > 0:
          boxes, probs = boxes_to_array(boxes_)
          boxes = to_minmax(boxes)
          labels = np.array([b.get_label() for b in boxes_])
          boxes = boxes[probs >= cls_threshold]
          labels = labels[probs >= cls_threshold]
          probs = probs[probs >= cls_threshold]
        else:
          boxes, labels, probs = [], [], []
        print(time.time()-s)
        image = cv2.imread(_imgpath)
        image = image[:, :, ::-1]
        visualize_boxes(image, boxes, labels, probs, self.labels)
        plt.imshow(image)
        plt.show()
  def _valid_epoch(self):
    with self.testwriter.as_default():
      for i, (imgs,labels) in enumerate(self.test_dataloader):
        logits = self.model(imgs, training=False)
        self.test_acc.update_state(tf.argmax(logits, axis=1),labels)
      tf.summary.scalar('loss', self.test_loss.result(),step=self.global_iter)
      tf.summary.scalar('accuracy', self.test_acc.result(),step=self.global_iter)
      self.test_loss.reset_states()
      self.test_acc.reset_states()
  def _train_epoch(self):
    with self.trainwriter.as_default():
      for i, (img,*labels) in enumerate(self.train_dataloader):
        self.global_iter+=1
        loss=self.train_step(img,labels)
        print(loss)
        if self.global_iter%self.log_iter==0:
          tf.summary.scalar('loss',self.train_loss.result(),step=self.global_iter)
          tf.summary.scalar('accuracy',self.train_acc.result(),step=self.global_iter)
          self.train_loss.reset_states()
          self.train_acc.reset_states()
if __name__ == '__main__':
  import os

  print(os.getcwd())
