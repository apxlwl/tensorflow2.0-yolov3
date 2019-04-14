import tensorflow as tf
import numpy as np
from utils.dataset_util import get_filelists, PascalVocXmlParser
import cv2
from dataset import transform
import tensorflow as tf
import os
from utils.dataset_util import DataGenerator

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PascalDataset:
  def __init__(self, configs,transform):
    dataset_dir=configs['dataset_dir']
    ann_dir=os.path.join(dataset_dir,configs['subset'],"annotations")
    self.anns = get_filelists(ann_dir, "*", "xml")
    self.labels = configs['labels']
    self.anchors = np.array(configs['anchors'])
    self._transform=transform

  def __len__(self):
    return len(self.anns)

  def __getitem__(self, idx):
    fname, bboxes, labels = PascalVocXmlParser(self.anns[idx], self.labels).parse()
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_shape = img.shape[:2]

    # Load the annotation.
    img,bboxes=self._transform(img,bboxes)
    list_grids = transform.preprocess(bboxes, labels, img.shape[:2], class_num=1, anchors=self.anchors)

    pad_scale=(1,1)
    return img.astype(np.float32), \
           fname, \
           np.array(pad_scale).astype(np.float32), \
           np.array(ori_shape).astype(np.float32), \
           list_grids[0].astype(np.float32), \
           list_grids[1].astype(np.float32), \
           list_grids[2].astype(np.float32),


def get_dataset(config):
  config['subset']='val'
  datatransform = transform.YOLO3DefaultValTransform(height=416, width=416, mean=(0, 0, 0), std=(1, 1, 1))
  valset = PascalDataset(config, datatransform)
  generator = DataGenerator(valset)
  valset = tf.data.Dataset.from_generator(generator,
                                          ((tf.float32, tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                                            tf.float32)))
  valset = valset.batch(config['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)

  config["subset"] = 'train'
  datatransform = transform.YOLO3DefaultTrainTransform(height=416,width=416,mean=(0,0,0),std=(1,1,1))
  trainset = PascalDataset(config,datatransform)
  generator = DataGenerator(trainset,shuffle=True)
  trainset = tf.data.Dataset.from_generator(generator,
                                            ((tf.float32, tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                                              tf.float32)))
  #be careful to drop the last smaller batch if using tf.function
  trainset = trainset.batch(config['batch_size'],drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return trainset, valset


if __name__ == '__main__':
  import json
  import matplotlib.pyplot as plt
  with open('../configs/face.json', 'r') as f:
    configs = json.load(f)
  configs['dataset']['batch_size']=2
  train,_ = get_dataset(configs['dataset'])
  for idx, inputs in enumerate(train):
    pass
