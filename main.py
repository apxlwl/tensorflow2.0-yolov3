from options import Options
from yolo.net.yolonet import Yolonet
from trainers.trainer import Trainer
from tensorflow import keras
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

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
  boundaries=[400000,800000],
  values=[args.learning_rate, args.learning_rate * 0.1, args.learning_rate * 0.01]
)

optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate,
                           momentum=args.momentum)

_Trainer = Trainer(args=args,
                   config=configs,
                   model=net,
                   optimizer=optimizer
                   )
if args.do_test:
  _Trainer._valid_epoch()
else:
  _Trainer.train()


  #

