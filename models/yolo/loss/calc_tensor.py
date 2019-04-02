# -*- coding: utf-8 -*-

import tensorflow as tf
from models.yolo.loss.utils import adjust_pred_tensor, adjust_true_tensor
from models.yolo.loss.utils import conf_delta_tensor
from models.yolo.loss.utils import loss_class_tensor, loss_conf_tensor, loss_coord_tensor, wh_scale_tensor
from tensorflow.python import keras


def sum_loss(losses):
  return tf.sqrt(tf.reduce_sum(losses))

def loss_fn(list_y_trues, list_y_preds,
            anchors,
            image_size,
            ignore_thresh=0.5,
            grid_scale=1,
            obj_scale=5,
            noobj_scale=1,
            xywh_scale=1,
            class_scale=1):
  if image_size is None:
    image_size = [288, 288]
  calculator = LossTensorCalculator(image_size=image_size,
                                    ignore_thresh=ignore_thresh,
                                    grid_scale=grid_scale,
                                    obj_scale=obj_scale,
                                    noobj_scale=noobj_scale,
                                    xywh_scale=xywh_scale,
                                    class_scale=class_scale)
  loss_box1,loss_conf1,loss_class1 = calculator.run(list_y_trues[0], list_y_preds[0], anchors=anchors[12:]) #feature5
  loss_box2,loss_conf2,loss_class2 = calculator.run(list_y_trues[1], list_y_preds[1], anchors=anchors[6:12])#feature4
  loss_box3,loss_conf3,loss_class3 = calculator.run(list_y_trues[2], list_y_preds[2], anchors=anchors[:6])#feature3
  loss_box=tf.reduce_sum(loss_box1+loss_box2+loss_box3)
  loss_conf=tf.reduce_sum(loss_conf1+loss_conf2+loss_conf3)
  loss_class=tf.reduce_sum(loss_class1+loss_class2+loss_class3)
  return loss_box,loss_conf,loss_class


class LossTensorCalculator(object):
  def __init__(self,
               image_size=[288, 288],
               ignore_thresh=0.5,
               grid_scale=1,
               obj_scale=5,
               noobj_scale=1,
               xywh_scale=1,
               class_scale=1):
    self.ignore_thresh = ignore_thresh
    self.grid_scale = grid_scale
    self.obj_scale = obj_scale
    self.noobj_scale = noobj_scale
    self.xywh_scale = xywh_scale
    self.class_scale = class_scale
    self.image_size = image_size  # (h, w)-ordered

  def run(self, y_true, y_pred, anchors):
    # 1. setup

    y_pred = tf.reshape(y_pred, y_true.shape)
    object_mask = y_true[..., 4:5]

    # print(object_mask.shape) (2, 9, 9, 3, 1)
    #the value is one for the true sample

    # 2. Adjust prediction (bxy, twh)
    preds = adjust_pred_tensor(y_pred)
    # print(preds.shape) (2, 9, 9, 3, 6)
    # 3. Adjust ground truth (bxy, twh)
    trues = adjust_true_tensor(y_true)
    # print(trues.shape) (2, 9, 9, 3, 6) #the last channel in the -1th dimension is class index
    # 4. conf_delta tensor
    conf_delta = conf_delta_tensor(y_true, preds, anchors, self.ignore_thresh)
    # print(conf_delta.shape) (2, 9, 9, 3)
    # 5. loss tensor
    #TODO readjust the imagesize with padded imageMeta
    wh_scale = wh_scale_tensor(trues[..., 2:4], anchors, self.image_size)
    # print(wh_scale.shape) (2, 9, 9, 3, 1)
    loss_box = loss_coord_tensor(object_mask, preds[..., :4], trues[..., :4], wh_scale, self.xywh_scale)
    loss_conf = loss_conf_tensor(object_mask, preds[..., 4:5], trues[..., 4:5], self.obj_scale, self.noobj_scale,conf_delta)
    loss_class = loss_class_tensor(object_mask, preds[..., 5:], trues[..., 5], self.class_scale)
    # loss = loss_box + loss_conf + loss_class
    return loss_box*self.grid_scale,loss_conf*self.grid_scale,loss_class*self.grid_scale
    # return loss * self.grid_scale


