from utils.util import ensure_dir
import tensorflow as tf
from tensorflow import summary
from dataset.coco import get_dataset
import numpy as np
import os
import time
class BaseTrainer:
  """
  Base class for all trainers
  """
  def __init__(self, args, configs, model, optimizer, scheduler):
    self.args = args
    self.configs = configs

    self.model = model
    self.optimizer = optimizer
    self.scheduler_config = scheduler
    self.scheduler = None
    self.experiment_name = args.experiment_name
    self.global_iter = tf.Variable(0)
    self.global_epoch = tf.Variable(0)
    self.train_dataloader = None
    self.test_dataloader = None
    self.log_iter = self.args.log_iter
    self.evaluate = self.args.evaluate
    self.anchors = np.array(self.configs["model"]["anchors"])
    self.net_size = self.configs["model"]["net_size"]
    self.labels=self.configs['model']['labels']

    self._get_model()
    self._get_SummaryWriter()
    self._get_dataset()
    self._get_scheduler()
    self._get_loggers()
    self._get_checkpoint()
  def is_better(self, new, old):
    pass

  def _get_checkpoint(self):
    self.ckpt = tf.train.Checkpoint(step=self.global_iter,epoch=self.global_epoch, optimizer=self.optimizer, net=self.model)
    self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.save_path, max_to_keep=10)
    if self.args.resume:
      self._load_checkpoint()

  def _load_checkpoint(self):
    # TODO add a dummy step
    if self.args.resume=="load_darknet":
      self.model.load_darknet_params(os.path.join(self.configs["pretrained_model"],
                                                  'darknet53.conv.74'),skip_detect_layer=True,body=True)
    elif self.args.resume=="load_yolov3":
      self.model.load_darknet_params(os.path.join(self.configs["pretrained_model"],
                                                  'yolov3.weights'), skip_detect_layer=False, body=False)
    else:
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
      self.global_iter=self.ckpt.step
      self.global_epoch=self.ckpt.epoch

    print("successfully load checkpoint {}".format(self.args.resume))

  def _get_scheduler(self):
    #TODO : add learningrate scheduler
    pass

  def _get_model(self):
    self.save_path = './checkpoints/{}/'.format(self.args.experiment_name)
    ensure_dir(self.save_path)
    self._prepare_device()
  def _prepare_device(self):
    #TODO: add distributed training
    pass
  def _get_SummaryWriter(self):
    if not self.args.debug and not self.args.evaluate:
      ensure_dir(os.path.join('./summary/',self.experiment_name))
      self.trainwriter = summary.create_file_writer(logdir='./summary/{}/{}/train'.format(self.experiment_name,
                                                                                          time.strftime(
                                                                                            "%m%d-%H-%M-%S",
                                                                                            time.localtime(
                                                                                              time.time()))))
  def _get_dataset(self):
    self.train_dataloader, self.test_dataloader = get_dataset(self.configs['dataset'])

  def _get_loggers(self):
    raise NotImplementedError

  def train(self):
    for epoch in range(self.global_epoch.numpy(), self.args.total_epoch):
      self.global_epoch.assign_add(1)
      self._train_epoch()

  @tf.function
  def train_step(self, inputs, gts):
    raise NotImplementedError

  def _train_epoch(self):
    raise NotImplementedError

  def _valid_epoch(self):
    raise NotImplementedError
