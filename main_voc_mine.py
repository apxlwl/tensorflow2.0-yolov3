from options import Options
from yolo.net.yolonet import Yolonet
from trainers.trainer_voc import Trainer
import tensorflow.python.keras as keras
from tensorflow import keras
import tensorflow as tf
import json
import os

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

opt = Options()
args = opt.opt
args.experiment_name = 'voc_scrach'
args.dataset_name='VOC'
args.dataset_root='/home/gwl/datasets/VOCdevkit'
args.lr_initial = 1e-4
args.config_path = './configs/voc.json'
args.total_epoch = 150
args.log_iter = 5000
args.batch_size = 6
args.net_size=544
# args.resume = 'load_darknet'
args.resume = 145
args.do_test = True

net = Yolonet(n_classes=20)

lr_schedule = keras.experimental.CosineDecay(
  initial_learning_rate=args.lr_initial,
  decay_steps=args.total_epoch*1380,
  alpha=0.01
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                 momentum=args.momentum)

_Trainer = Trainer(args=args,
                   model=net,
                   optimizer=optimizer,
                   )
if args.do_test:
  _Trainer._valid_epoch()
else:
  _Trainer.train()

  #
