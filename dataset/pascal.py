import tensorflow as tf
import numpy as np
from utils.dataset_util import get_filelists, PascalVocXmlParser
import cv2
from dataset import transform
import tensorflow as tf
import os
from utils.dataset_util import DataGenerator
from utils.visualize import visualize_boxes
from base import VOC_LABEL,VOC_ANCHOR

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class VocDataset:
  def __init__(self, dataset_root,transform,subset,shuffle):
    self.dataset_root = dataset_root
    self.labels = VOC_LABEL
    self.anchors = np.array(VOC_ANCHOR)
    self._transform = transform
    self._annopath = os.path.join('{}', 'Annotations', '{}.xml')
    self._imgpath = os.path.join('{}', 'JPEGImages', '{}.jpg')
    self._ids = []
    self.shuffle = shuffle
    for year, set in subset:
      rootpath = os.path.join(dataset_root, 'VOC' + year)
      for line in open(os.path.join(rootpath, 'ImageSets', 'Main', '{}.txt'.format(set))):
        self._ids.append((rootpath, line.strip()))

  def __len__(self):
    return len(self._ids)

  def __call__(self):
    indices = np.arange(len(self._ids))
    if self.shuffle:
      np.random.shuffle(indices)
    for idx in indices:
      rootpath, filename = self._ids[idx]
      annpath = self._annopath.format(rootpath, filename)
      imgpath = self._imgpath.format(rootpath, filename)
      fname, bboxes, labels,_ = PascalVocXmlParser(annpath, self.labels).parse()
      img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ori_shape = img.shape[:2]
      # Load the annotation.
      img, bboxes = self._transform(img, bboxes)
      list_grids = transform.preprocess(bboxes, labels, img.shape[:2], class_num=len(self.labels), anchors=self.anchors)
      pad_scale = (1, 1)
      yield img.astype(np.float32), \
            imgpath, \
            annpath, \
            np.array(pad_scale).astype(np.float32), \
            np.array(ori_shape).astype(np.float32), \
            list_grids[0].astype(np.float32), \
            list_grids[1].astype(np.float32), \
            list_grids[2].astype(np.float32), \

def get_dataset(dataset_root,batch_size):
  subset = [('2007', 'test')]
  datatransform = transform.YOLO3DefaultValTransform(height=416, width=416, mean=(0, 0, 0), std=(1, 1, 1))
  valset = VocDataset(dataset_root, datatransform,subset,shuffle=False)

  valset_iter = tf.data.Dataset.from_generator(valset,
                                          ((tf.float32, tf.string,tf.string, tf.float32, tf.float32,
                                            tf.float32, tf.float32,tf.float32)))
  valset_iter = valset_iter.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

  subset = [('2007', 'trainval'), ('2012', 'trainval')]
  datatransform = transform.YOLO3DefaultTrainTransform(height=416, width=416, mean=(0, 0, 0), std=(1, 1, 1))
  trainset = VocDataset(dataset_root, datatransform,subset,shuffle=True)
  trainset_iter = tf.data.Dataset.from_generator(trainset,
                                            ((tf.float32, tf.string,tf.string, tf.float32, tf.float32,
                                              tf.float32, tf.float32,tf.float32)))
  # be careful to drop the last smaller batch if using tf.function
  trainset_iter = trainset_iter.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return trainset_iter, valset_iter


if __name__ == '__main__':
  import json
  import matplotlib.pyplot as plt

  with open('../configs/voc.json', 'r') as f:
    configs = json.load(f)
  configs['dataset']['batch_size'] = 2
  configs['dataset']['dataset_dir']='/disk3/datasets/voc'
  train, _ = get_dataset(configs['dataset'])
  assert 0
  for epoch in range(5):
    for idx, inputs in enumerate(train):
      img=inputs[0]
      for im in img:
        print(im.shape)
        plt.imshow(im)
        plt.show()