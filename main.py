from options import Options
from yolo.net.yolonet import Yolonet
from trainers.trainer import Trainer
import tensorflow.python.keras as keras
from tensorflow.keras import optimizers
import tensorflow as tf
import json
import os
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

opt = Options()
args = opt.opt
args.experiment_name = 'yolov3'

with open(args.config_path,'r') as f:
  configs = json.load(f)

net = Yolonet(n_classes=80,freeze_backbone=args.freeze_darknet)

optimizer = optimizers.SGD(learning_rate=args.learning_rate,
                           momentum=args.momentum)
scheduler=None

_Trainer = Trainer(args=args,
                   config=configs,
                   model=net,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   )
if args.do_test:
  _Trainer._valid_epoch()
else:
  _Trainer.train()


  #

