from utils.util import ensure_dir
import tensorflow as tf
from tensorflow import summary
from dataset import get_COCO, get_VOC
import os
import time
from config import *
from tensorflow.python.keras import metrics
from dataset import makeImgPyramids
from yolo import predict_yolo,loss_yolo
from utils.nms_utils import gpu_nms
import numpy as np
from dataset import bbox_flip
from  collections import defaultdict
import shutil

class BaseTrainer:
  """
  Base class for all trainers
  """

  def __init__(self, args, model, optimizer):
    self.args = args
    self.model = model
    self.optimizer = optimizer
    self.experiment_name = args.experiment_name
    self.dataset_name = args.dataset_name
    self.dataset_root = args.dataset_root
    self.batch_size = args.batch_size

    self.train_dataloader = None
    self.test_dataloader = None
    self.log_iter = self.args.log_iter
    self.net_size = self.args.net_size
    self.anchors = eval('{}_ANCHOR_{}'.format(self.args.dataset_name.upper(),self.net_size))
    self.labels = eval('{}_LABEL'.format(self.args.dataset_name.upper()))
    self.num_classes = len(self.labels)

    #logger attributes
    self.global_iter = tf.Variable(0)
    self.global_epoch = tf.Variable(0)
    self.TESTevaluator=None
    self.LossBox = None
    self.LossConf = None
    self.LossClass = None
    self.logger_custom=None
    self.metric_evaluate=None

    #initialize
    self._get_model()
    self._get_SummaryWriter()
    self._get_checkpoint()
    self._get_dataset()
    self._get_loggers()

  def _get_checkpoint(self):
    self.ckpt = tf.train.Checkpoint(step=self.global_iter, epoch=self.global_epoch, optimizer=self.optimizer,
                                    net=self.model)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.save_path, max_to_keep=5)
    if self.args.resume:
      self._load_checkpoint()

  def _load_checkpoint(self):
    if self.args.resume == "load_darknet":
      self.model.load_darknet_params(os.path.join(self.args.pretrained_model,
                                                  'darknet53.conv.74'), skip_detect_layer=True, body=True)
    elif self.args.resume == "load_yolov3":
      self.model.load_darknet_params(os.path.join(self.args.pretrained_model,
                                                  'yolov3.weights'), skip_detect_layer=False, body=False)
    else:
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
      self.ckpt.restore(os.path.join(self.save_path, 'ckpt-{}'.format(self.args.resume)))
      self.global_iter = self.ckpt.step
      self.global_epoch = self.ckpt.epoch
    print("successfully load checkpoint {}".format(self.args.resume))

  def _get_model(self):
    self.save_path = './checkpoints/{}/'.format(self.args.experiment_name)
    ensure_dir(self.save_path)
    self._prepare_device()

  def _prepare_device(self):
    # TODO: add distributed training
    pass

  def _get_SummaryWriter(self):
    if not self.args.debug and not self.args.do_test:
      ensure_dir(os.path.join('./summary/', self.experiment_name))
      self.summarywriter = summary.create_file_writer(logdir='./summary/{}/{}/train'.format(self.experiment_name,
                                                                                          time.strftime(
                                                                                            "%m%d-%H-%M-%S",
                                                                                            time.localtime(
                                                                                              time.time()))))

  def _get_dataset(self):
    self.train_dataloader, self.test_dataloader = eval('get_{}'.format(self.dataset_name))(
      dataset_root=self.dataset_root,
      batch_size=self.args.batch_size,
      net_size=self.net_size
    )

  def train(self):
    best_mAP=-1
    for epoch in range(self.global_epoch.numpy(), self.args.total_epoch):
      self.global_epoch.assign_add(1)
      self._train_epoch()
      results, imgs = self._valid_epoch(multiscale=False, flip=False)
      with self.summarywriter.as_default():
        current_lr = self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
        tf.summary.scalar("learning_rate", current_lr, step=self.global_iter.numpy())
        for k, v in zip(self.logger_custom, results):
          tf.summary.scalar(k, v, step=self.global_iter.numpy())
        for k, v in self.logger_losses.items():
          tf.summary.scalar(k, v.result(), step=self.global_iter.numpy())
        for i in range(len(imgs)):
          tf.summary.image("detections_{}".format(i), tf.expand_dims(tf.convert_to_tensor(imgs[i]), 0),
                           step=self.global_iter.numpy())
      self._reset_loggers()
      self.ckpt_manager.save(self.global_epoch)
      if results[0]>best_mAP:
        shutil.copyfile(
          os.path.join(self.save_path, 'ckpt-{}.index'.format(self.global_epoch.numpy())),
          os.path.join(self.save_path, 'ckpt-{}.index'.format("best"))
        )
        shutil.copyfile(
          os.path.join(self.save_path, 'ckpt-{}.data-00000-of-00001'.format(self.global_epoch.numpy())),
          os.path.join(self.save_path, 'ckpt-{}.data-00000-of-00001'.format("best"))
        )
  def _get_loggers(self):
    self.LossBox = metrics.Mean()
    self.LossConf = metrics.Mean()
    self.LossClass = metrics.Mean()
    self.logger_losses = {}
    self.logger_losses.update({"lossBox": self.LossBox})
    self.logger_losses.update({"lossConf": self.LossConf})
    self.logger_losses.update({"lossClass": self.LossClass})

  def _reset_loggers(self):
    self.TESTevaluator.reset()
    self.LossClass.reset_states()
    self.LossConf.reset_states()
    self.LossBox.reset_states()

  # @tf.function
  def train_step(self, imgs, labels):
    with tf.GradientTape() as tape:
      outputs = self.model(imgs, training=True)
      loss_box, loss_conf, loss_class = loss_yolo(outputs, labels, anchors=self.anchors,
                                                  inputshape=(self.net_size, self.net_size),
                                                  num_classes=self.num_classes)
      loss = tf.reduce_sum(loss_box + loss_conf + loss_class)
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    self.LossBox.update_state(loss_box)
    self.LossConf.update_state(loss_conf)
    self.LossClass.update_state(loss_class)

  def _train_epoch(self):
    for i, inputs in enumerate(self.train_dataloader):
      inputs = [tf.squeeze(input, axis=0) for input in inputs]
      img, _, _, _, _, *labels = inputs
      self.global_iter.assign_add(1)
      if self.global_iter.numpy() % 200 == 0:
        tf.print(self.global_iter.numpy())
        for k, v in self.logger_losses.items():
          tf.print(k, ":", v.result().numpy())
      self.train_step(img, labels)

  def _valid_epoch(self,multiscale,flip):
    for idx_batch, inputs in enumerate(self.test_dataloader):
      if idx_batch == self.args.valid_batch and not self.args.do_test:  # to save time
        break
      if idx_batch==5:
        break
      inputs = [tf.squeeze(input, axis=0) for input in inputs]
      (imgs, imgpath, annpath, padscale, ori_shapes, *_) = inputs
      if not multiscale:
        INPUT_SIZES = [self.net_size]
      else:
        INPUT_SIZES = [self.net_size-32,self.net_size,self.net_size+32]
      pyramids = makeImgPyramids(imgs.numpy(), scales=INPUT_SIZES, flip=flip)

      # produce outputFeatures for each scale
      img2multi = defaultdict(list)
      for idx, pyramid in enumerate(pyramids):
        grids = self.model(pyramid, training=False)
        for imgidx in range(imgs.shape[0]):
          img2multi[imgidx].append([grid[imgidx] for grid in grids])

      # append prediction for each image per scale/flip
      for imgidx, scalegrids in img2multi.items():
        allboxes = []
        allscores = []
        for _grids, _scale in zip(scalegrids[:len(INPUT_SIZES)], INPUT_SIZES):
          _boxes, _scores = predict_yolo(_grids, self.anchors, (_scale, _scale), ori_shapes[imgidx],
                                         padscale=padscale[imgidx], num_classes=self.num_classes)
          allboxes.append(_boxes)
          allscores.append(_scores)
        if flip:
          for _grids, _scale in zip(scalegrids[len(INPUT_SIZES):], INPUT_SIZES):
            _boxes, _scores = predict_yolo(_grids, self.anchors, (_scale, _scale), ori_shapes[imgidx],
                                           padscale=padscale[imgidx], num_classes=self.num_classes)
            _boxes = bbox_flip(tf.squeeze(_boxes).numpy(), flip_x=True, size=ori_shapes[imgidx][::-1])
            _boxes = _boxes[np.newaxis, :]
            allboxes.append(_boxes)
            allscores.append(_scores)
        # TODO change nms input to y1x1y2x2
        nms_boxes, nms_scores, nms_labels = gpu_nms(tf.concat(allboxes, axis=1), tf.concat(allscores, axis=1),
                                                    num_classes=self.num_classes)
        self.TESTevaluator.append(imgpath[imgidx].numpy(),
                                  annpath[imgidx].numpy(),
                                  nms_boxes.numpy(),
                                  nms_scores.numpy(),
                                  nms_labels.numpy())
    results = self.TESTevaluator.evaluate()
    imgs = self.TESTevaluator.visual_imgs
    for k, v in zip(self.logger_custom, results):
      print("{}:{}".format(k, v))
    return results, imgs
