from options import Options
from yolo.net.yolonet import Yolonet
from trainers.trainer_voc import Trainer
from tensorflow import keras
import tensorflow as tf
import json
import os
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

opt = Options()
args = opt.opt
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.experiment_name = 'yolov3'
net = Yolonet(n_classes=20,freeze_backbone=args.freeze_darknet)

#the total size of your dataset
lensVOC=16551
lr_schedule = keras.experimental.CosineDecay(
  initial_learning_rate=args.lr_initial,
  decay_steps=args.total_epoch*(lensVOC//args.batch_size),
  alpha=0.01
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                 momentum=args.momentum)

_Trainer = Trainer(args=args,
                   model=net,
                   optimizer=optimizer,
                   )
if args.do_test:
  _Trainer._valid_epoch(args.multitest,args.fliptest)
else:
  _Trainer.train()


  #

