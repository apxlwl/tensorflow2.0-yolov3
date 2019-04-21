from base.base_trainer import BaseTrainer
import tensorflow as tf
from evaluator.voceval import EvaluatorVOC
from  collections import defaultdict
from tensorflow.python.keras import metrics
from yolo.yolo_loss import loss_yolo
import time
from dataset import makeImgPyramids
from base import TEST_INPUT_SIZES
from yolo import predict_yolo
from utils.nms_utils import gpu_nms
import numpy as np
from dataset import bbox_flip
import matplotlib.pyplot as plt
class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer):
    super().__init__(args, model, optimizer)

  def _get_loggers(self):
    self.TESTevaluator = EvaluatorVOC(anchors=self.anchors,
                                       cateNames=self.labels,
                                       rootpath=self.dataset_root,
                                       score_thres=0.01,
                                       iou_thres=0.5,
                                      use_07_metric=False
                                      )

    self.LossBox = metrics.Mean()
    self.LossConf = metrics.Mean()
    self.LossClass = metrics.Mean()
    self.logger_losses = {}
    self.logger_losses.update({"lossBox": self.LossBox})
    self.logger_losses.update({"lossConf": self.LossConf})
    self.logger_losses.update({"lossClass": self.LossClass})
    self.logger_voc = ['AP@{}'.format(cls) for cls in self.labels] + ['mAP']

  def _reset_loggers(self):
    self.TESTevaluator.reset()
    self.LossClass.reset_states()
    self.LossConf.reset_states()
    self.LossBox.reset_states()

  @tf.function
  def train_step(self, imgs, labels):
    with tf.GradientTape() as tape:
      outputs = self.model(imgs, training=True)
      inputshape=tf.shape(imgs)[1:3]
      loss_box, loss_conf, loss_class = loss_yolo(outputs, labels, anchors=self.anchors,
                                                  inputshape=inputshape,
                                                  num_classes=self.num_classes)
      loss = tf.reduce_sum(loss_box + loss_conf + loss_class)
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    self.LossBox.update_state(loss_box)
    self.LossConf.update_state(loss_conf)
    self.LossClass.update_state(loss_class)
    return outputs

  def _valid_epoch(self,multiscale=False,flip=False):
    s=time.time()
    for idx_batch, inputs in enumerate(self.test_dataloader):
      if idx_batch == self.args.valid_batch and not self.args.do_test:  # to save time
        break
      inputs = [tf.squeeze(input, axis=0) for input in inputs]
      (imgs, imgpath, annpath, padscale, ori_shapes, *_)=inputs
      if not multiscale:
        INPUT_SIZES=[self.net_size]
      else:
        INPUT_SIZES=TEST_INPUT_SIZES
      pyramids=makeImgPyramids(imgs.numpy(),scales=INPUT_SIZES,flip=flip)

      #produce outputFeatures for each scale
      img2multi=defaultdict(list)
      for idx,pyramid in enumerate(pyramids):
        grids = self.model(pyramid, training=False)
        for imgidx in range(imgs.shape[0]):
          img2multi[imgidx].append([grid[imgidx] for grid in grids])

      #append prediction for each image per scale/flip
      for imgidx,scalegrids in img2multi.items():
        allboxes=[]
        allscores=[]
        for _grids,_scale in zip(scalegrids[:len(INPUT_SIZES)],TEST_INPUT_SIZES):
          _boxes, _scores = predict_yolo(_grids, self.anchors, (_scale,_scale), ori_shapes[imgidx],
                                         padscale=padscale[imgidx], num_classes=20)
          allboxes.append(_boxes)
          allscores.append(_scores)
        if flip:
          for _grids, _scale in zip(scalegrids[len(INPUT_SIZES):], INPUT_SIZES):
            _boxes, _scores = predict_yolo(_grids, self.anchors, (_scale, _scale), ori_shapes[imgidx],
                                           padscale=padscale[imgidx], num_classes=20)
            _boxes = bbox_flip(tf.squeeze(_boxes).numpy(), flip_x=True, size=ori_shapes[imgidx][::-1])
            _boxes = _boxes[np.newaxis,:]
            allboxes.append(_boxes)
            allscores.append(_scores)
        #TODO change nms input to y1x1y2x2
        nms_boxes, nms_scores, nms_labels=gpu_nms(tf.concat(allboxes,axis=1),tf.concat(allscores,axis=1),num_classes=self.num_classes)
        self.TESTevaluator.append(imgpath[imgidx].numpy(),
                                  annpath[imgidx].numpy(),
                                  nms_boxes.numpy(),
                                  nms_scores.numpy(),
                                  nms_labels.numpy())
    results = self.TESTevaluator.evaluate()
    imgs = self.TESTevaluator.visual_imgs
    print(time.time()-s)
    return results, imgs

  def _train_epoch(self):
    for i, inputs in enumerate(self.train_dataloader):
      inputs = [tf.squeeze(input, axis=0) for input in inputs]
      img, _, _, _, _, *labels=inputs
      self.global_iter.assign_add(1)
      if self.global_iter.numpy() % 100 == 0:
        print(self.global_iter.numpy())
        for k, v in self.logger_losses.items():
          print(k, ":", v.result().numpy())
      _ = self.train_step(img, labels)

      if self.global_iter.numpy() % self.log_iter == 0:
        results, imgs = self._valid_epoch()
        with self.trainwriter.as_default():
          current_lr= self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
          tf.summary.scalar("learning_rate", current_lr, step=self.global_iter.numpy())

          for k, v in zip(self.logger_voc, results):
            tf.summary.scalar(k, v, step=self.global_iter.numpy())
          for k, v in self.logger_losses.items():
            tf.summary.scalar(k, v.result(), step=self.global_iter.numpy())
          for i in range(len(imgs)):
            tf.summary.image("detections_{}".format(i), tf.expand_dims(tf.convert_to_tensor(imgs[i]), 0),
                             step=self.global_iter.numpy())
        self._reset_loggers()
    self.ckpt_manager.save(self.global_epoch)
