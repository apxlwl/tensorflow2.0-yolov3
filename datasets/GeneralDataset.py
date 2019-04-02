import tensorflow as tf
import numpy as np
from utils.dataset_util import get_filelists, PascalVocXmlParser,_create_empty_grid,_assign_box,_encode_box
from utils.box_util import find_match_anchor, create_anchor_boxes
import cv2
import datasets.transform as transforms
import os
from datasets import COCO_ANCHORS

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DOWNSAMPLE_RATIO = 32
DEFAULT_NETWORK_SIZE = 288


class PascalDataset:
  def __init__(self, configs):
    self.anns = get_filelists(configs['train_anns'], "*", "xml")
    self.labels = configs['labels']
    self.anchors = create_anchor_boxes(COCO_ANCHORS)
    # mean = (123.675, 116.28, 103.53)
    mean = (0, 0, 0)
    std = (1., 1., 1.)
    self.img_transform = transforms.ImageTransform((DEFAULT_NETWORK_SIZE, DEFAULT_NETWORK_SIZE),
                                                   mean, std, 'fixed')
    self.bbox_transform = transforms.BboxTransform()
  def __len__(self):
    return len(self.anns)

  def __getitem__(self, idx):
    fname, boxes, labels = PascalVocXmlParser(self.anns[idx], self.labels).parse()
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flip = True if np.random.rand() < 0.5 else False
    img, img_shape, scale_factor = self.img_transform(img, flip)
    boxes, labels = self.bbox_transform(
      boxes, labels, img_shape, scale_factor, flip)

    list_grids = _create_empty_grid(DEFAULT_NETWORK_SIZE, n_classes=1)
    for box, label in zip(boxes, labels):
      match_anchor, scale_index, box_index = find_match_anchor(box, self.anchors)
      coded_box = _encode_box(list_grids[scale_index], box, match_anchor, DEFAULT_NETWORK_SIZE, DEFAULT_NETWORK_SIZE)
      _assign_box(list_grids[scale_index], box_index, coded_box, label)
    return img.astype(np.float32), \
           list_grids[0].astype(np.float32), \
           list_grids[1].astype(np.float32), \
           list_grids[2].astype(np.float32)


class DataGenerator:
  def __init__(self, dataset, shuffle=False):
    self.dataset = dataset
    self.shuffle = shuffle

  def __call__(self):
    indices = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.shuffle(indices)
    for img_idx in indices:
      img,label0,label1,label2 = self.dataset[img_idx]
      yield img,label0,label1,label2


def get_dataset(config):
  trainset = PascalDataset(config)
  generator = DataGenerator(trainset)
  tf_dataset = tf.data.Dataset.from_generator(generator, ((tf.float32, tf.float32, tf.float32, tf.float32)))
  tf_dataset = tf_dataset.batch(config['batch_size'])
  return tf_dataset, tf_dataset


if __name__ == '__main__':
  import json
  from utils.visualize import draw_boxes
  import matplotlib.pyplot as plt

  with open('../configs/face.json', 'r') as f:
    configs = json.load(f)
  train,_ = get_dataset(configs['dataset'])
  for idx, inputs in enumerate(train):

    assert 0
