from options import Options
from yolo.net.yolonet import Yolonet
from trainers.trainer_custom import Trainer
import tensorflow.python.keras as keras
from tensorflow import keras
import tensorflow as tf
import json
import os

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

opt = Options()
args = opt.opt
args.experiment_name = 'face'
# args.learning_rate = 0.0005
args.config_path = './configs/face.json'
args.total_epoch = 80
args.log_iter = 9000
args.resume = 'load_darknet'
args.do_test = False
with open(args.config_path, 'r') as f:
  configs = json.load(f)
net = Yolonet(n_classes=1)
# lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
#   boundaries=[50,100],
#   values=[args.learning_rate, args.learning_rate * 0.1, args.learning_rate * 0.01]
# )
optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate,
                                 momentum=args.momentum)

_Trainer = Trainer(args=args,
                   config=configs,
                   model=net,
                   optimizer=optimizer,
                   )
if args.do_test:
  _Trainer._valid_epoch()
else:
  _Trainer.train()

  #
