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

opt = Options()
args = opt.opt
args.experiment_name = 'test'
args.gpu='1'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
args.dataset_name='VOC'
args.dataset_root='/home/gwl/datasets/VOCdevkit'
args.lr_initial = 1e-4
args.total_epoch = 250
args.log_iter = 5000
args.batch_size = 12
args.net_size= 480
args.fliptest=False
args.multitest=False
args.resume = 'load_darknet'
# args.resume = 182
# args.do_test = True
lensVOC=16551
net = Yolonet(n_classes=20)

lr_schedule = keras.experimental.CosineDecay(
  initial_learning_rate=args.lr_initial,
  decay_steps=150*(lensVOC//args.batch_size),
  alpha=0.01
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                 momentum=args.momentum)

_Trainer = Trainer(args=args,
                   model=net,
                   optimizer=optimizer,
                   )
if args.do_test:
  _Trainer._valid_epoch(multiscale=args.multitest,flip=args.fliptest)
else:
  _Trainer.train()

  #
