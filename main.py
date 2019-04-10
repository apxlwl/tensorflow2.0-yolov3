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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
opt = Options(None)
args = opt.opt
args.experiment_name = 'darknet_new'
args.gpu_ids = ["/gpu:0", "/gpu:1"]
args.learning_rate = 0.002
args.config_path='./configs/coco.json'
# args.resume= 'load_darknet'
args.resume = 4
args.total_epoch=30
args.log_iter=5000
args.freeze_darknet=True

with open(args.config_path,'r') as f:
  configs = json.load(f)
# --------------criterion------------
criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# --------------criterion------------

net = Yolonet(n_classes=80)
#for op in net.body.layers:
#  op.trainable = False
optimizer = optimizers.SGD(learning_rate=args.learning_rate, momentum=0.9)


# # --------------tune the learning rate---------------
scheduler=None
_Trainer = Trainer(args=args,
                   config=configs,
                   model=net,
                   criterion=criterion,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   )
_Trainer.train()
#
# _Trainer._valid_epoch()

# if (args.evaluate):
#   _Trainer._valid_epoch()
# else:
#   _Trainer.train()

  #

