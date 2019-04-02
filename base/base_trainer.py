from utils.util import ensure_dir
import tensorflow as tf
from tensorflow import summary
import json
# from datasets.GeneralDataset import get_dataset
from datasets.coco import get_dataset
import numpy as np
import numbers
import os
import time
from collections import OrderedDict
from utils.optimizer_util import load_opti, save_opti
from models.yolo.loss.newloss import LossCalculator
class BaseTrainer:
  """
  Base class for all trainers
  """

  def __init__(self, args, configs, model, criterions, optimizer, scheduler):
    self.args = args
    self.configs = configs

    self.model = model
    self.criterion = criterions
    self.optimizer = optimizer
    self.scheduler_config = scheduler
    self.scheduler = None
    self.experiment_name = args.experiment_name
    self.global_iter = 0
    self.global_epoch = 0
    self.visual_train = []
    self.visual_test = []
    self.train_dataloader = None
    self.test_dataloader = None
    self.save_iter = self.args.save_iter
    self.log_iter = self.args.log_iter
    self.evaluate = self.args.evaluate
    self.anchors = np.array(self.configs["model"]["anchors"])
    self.net_size = self.configs["model"]["net_size"]
    self.labels=self.configs['model']['labels']

    self._get_SummaryWriter()
    self._get_dataset()
    self._model_init()
    self._get_scheduler()
    self._get_loggers()
    self.lossfn=LossCalculator(anchors=self.configs['model']['anchors'],imgsize=(self.configs['model']['net_size'],
                                                                          self.configs['model']['net_size']))
  def is_better(self, new, old):
    pass

  def _save_checkpoint(self, name=None):
    if self.args.evaluate:
      return
    if name == None:
      self.model.save_weights(os.path.join(self.save_path, 'model-iter{}.h5'.format(self.global_iter)))
      save_opti(self.optimizer, os.path.join(self.save_path, 'opti-iter{}.pkl'.format(self.global_iter)))
    else:
      self.model.save_weights(os.path.join(self.save_path, 'model-{}.h5'.format(name)))
      save_opti(self.optimizer, os.path.join(self.save_path, 'opti-{}.pkl'.format(name)))
    print("save checkpoints at iter{}".format(self.global_iter))

  def _load_checkpoint(self):
    # TODO add a dummy step
    if self.args.resume=="load_darknet":
      self.model.load_darknet_params(os.path.join(self.configs["pretrained_model"],
                                                  'darknet53.conv.74'),skip_detect_layer=True,body=True)
    elif self.args.resume=="load_yolov3":
      self.model.load_darknet_params(os.path.join(self.configs["pretrained_model"],
                                                  'yolov3.weights'), skip_detect_layer=False, body=False)
    elif self.args.resume=="load_best":
      self.model.load_weights(os.path.join(self.save_path, 'model-best.h5'.format(self.args.resume)))
      # load_opti(self.optimizer, os.path.join(self.save_path, 'opti-best.pkl'.format(self.args.resume)))
    else:
      self.model.load_weights(os.path.join(self.save_path, 'model-iter{}.h5'.format(int(self.args.resume))))
      # load_opti(self.optimizer, os.path.join(self.save_path, 'opti-iter{}.pkl'.format(self.args.resume)))
    print("successfully load checkpoint {}".format(self.args.resume))

  def _get_scheduler(self):
    pass
    # if self.scheduler_config['lr_policy'] == 'cosin':
    #   self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_config['T_max'])
    #   self.scheduler_epoch = self.scheduler_config['decrease_Startepoch']
    # elif self.scheduler_config['lr_policy'] == 'step':
    #   self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.scheduler_config['gamma'])
    #   self.scheduler_epoch = self.scheduler_config['decrease_Startepoch']

  def _model_init(self):
    self.save_path = './checkpoints/{}/'.format(self.args.experiment_name)
    if self.args.resume:
      self._load_checkpoint()

  def _prepare_device(self):
    os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_id
    # return device, gpus

  def _get_SummaryWriter(self):
    self.save_path = './checkpoints/{}/'.format(self.args.experiment_name)
    ensure_dir(self.save_path)
    if not self.args.debug and not self.args.evaluate:
      # with open(self.save_path + "/args-{}.json".
      #     format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),'w') as f:
      #   json.dump(self.args.__dict__, f, indent=4)
      self.trainwriter = summary.create_file_writer(logdir='./summary/{}-{}/train'.format(self.experiment_name,
                                                                                          time.strftime(
                                                                                            "%m%d-%H-%M-%S",
                                                                                            time.localtime(
                                                                                              time.time()))))
      self.testwriter = summary.create_file_writer(logdir='./summary/{}-{}/test'.format(self.experiment_name,
                                                                                        time.strftime("%m%d-%H-%M-%S",
                                                                                                      time.localtime(
                                                                                                        time.time()))))

  def _get_dataset(self):
    self.train_dataloader, self.test_dataloader = get_dataset(self.configs['dataset'])

  def _get_loggers(self):
    raise NotImplementedError

  def train(self):
    for epoch in range(self.global_epoch, self.args.total_epoch):
      self.global_epoch += 1
      self._train_epoch()

  @tf.function
  def train_step(self, inputs, gts):
    raise NotImplementedError

  def _train_epoch(self):
    raise NotImplementedError

  def _valid_epoch(self):
    raise NotImplementedError
