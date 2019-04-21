import numpy as np
from utils.dataset_util import PascalVocXmlParser
import cv2
from dataset.augment import transform
import tensorflow as tf
import os
from config import VOC_LABEL,VOC_ANCHOR_512,TRAIN_INPUT_SIZES_VOC
import random
tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class VOCdataset:
  def __init__(self, dataset_root,transform,subset,batchsize,netsize,shuffle):
    self.dataset_root = dataset_root
    self.labels = VOC_LABEL
    self.anchors = np.array(eval("VOC_ANCHOR_{}".format(netsize)))
    self._transform = transform
    self._annopath = os.path.join('{}', 'Annotations', '{}.xml')
    self._imgpath = os.path.join('{}', 'JPEGImages', '{}.jpg')
    self._ids = []
    self.netsize=netsize
    self.batch_size=batchsize
    self.shuffle = shuffle
    self.multisizes=TRAIN_INPUT_SIZES_VOC
    for year, set in subset:
      rootpath = os.path.join(dataset_root, 'VOC' + year)
      for line in open(os.path.join(rootpath, 'ImageSets', 'Main', '{}.txt'.format(set))):
        self._ids.append((rootpath, line.strip()))

  def __len__(self):
    return len(self._ids)//self.batch_size

  def _load_batch(self,idx_batch,random_trainsize):
    img_batch = []
    imgpath_batch = []
    annpath_batch = []
    pad_scale_batch = []
    ori_shape_batch = []
    grid0_batch = []
    grid1_batch = []
    grid2_batch = []
    for idx in range(self.batch_size):
      rootpath, filename = self._ids[idx_batch * self.batch_size + idx]
      annpath = self._annopath.format(rootpath, filename)
      imgpath = self._imgpath.format(rootpath, filename)
      fname, bboxes, labels, _ = PascalVocXmlParser(annpath, self.labels).parse()
      img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ori_shape = img.shape[:2]
      # Load the annotation.
      img, bboxes = self._transform(random_trainsize, random_trainsize, img, bboxes)
      list_grids = transform.preprocess(bboxes, labels, img.shape[:2], class_num=len(self.labels), anchors=self.anchors)
      pad_scale = (1, 1)
      img_batch.append(img)
      imgpath_batch.append(imgpath)
      annpath_batch.append(annpath)
      ori_shape_batch.append(ori_shape)
      pad_scale_batch.append(pad_scale)
      grid0_batch.append(list_grids[0])
      grid1_batch.append(list_grids[1])
      grid2_batch.append(list_grids[2])
    return np.array(img_batch).astype(np.float32), \
           imgpath_batch, \
           annpath_batch, \
           np.array(pad_scale_batch).astype(np.float32), \
           np.array(ori_shape_batch).astype(np.float32), \
           np.array(grid0_batch).astype(np.float32), \
           np.array(grid1_batch).astype(np.float32), \
           np.array(grid2_batch).astype(np.float32)
  def __call__(self):
    indices = np.arange(len(self._ids))
    if self.shuffle:
      np.random.shuffle(indices)
    for idx_batch in range(self.__len__()):
      if self.shuffle:
        trainsize=random.choice(self.multisizes)
      else:
        trainsize =self.netsize
      yield self._load_batch(idx_batch,trainsize)



def get_dataset(dataset_root,batch_size,net_size):
  subset = [('2007', 'test')]
  datatransform = transform.YOLO3DefaultValTransform(mean=(0, 0, 0), std=(1, 1, 1))
  valset = VOCdataset(dataset_root, datatransform,subset,batch_size,net_size,shuffle=False)

  valset_iter = tf.data.Dataset.from_generator(valset,
                                          ((tf.float32, tf.string,tf.string, tf.float32, tf.float32,
                                            tf.float32, tf.float32,tf.float32)))
  valset_iter = valset_iter.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

  subset = [('2007', 'trainval'), ('2012', 'trainval')]
  datatransform = transform.YOLO3DefaultTrainTransform(mean=(0, 0, 0), std=(1, 1, 1))
  trainset = VOCdataset(dataset_root, datatransform,subset,batch_size,net_size,shuffle=True)
  trainset_iter = tf.data.Dataset.from_generator(trainset,
                                            ((tf.float32, tf.string,tf.string, tf.float32, tf.float32,
                                              tf.float32, tf.float32,tf.float32)))
  # be careful to drop the last smaller batch if using tf.function
  trainset_iter = trainset_iter.batch(1, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return trainset_iter, valset_iter


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  datatransform = transform.YOLO3DefaultTrainTransform(mean=(0, 0, 0), std=(1, 1, 1))
  subset = [('2007', 'trainval'), ('2012', 'trainval')]
  batch_size=2
  trainset = VOCdataset('/home/gwl/datasets/VOCdevkit', datatransform, subset, batch_size, shuffle=True)
  train, val = get_dataset('/home/gwl/datasets/VOCdevkit',8)
  for epoch in range(5):
    for idx, inputs in enumerate(val):
      inputs=[tf.squeeze(input,axis=0) for input in inputs]
      plt.imshow(inputs[0][0])
      plt.show()
      # assert 0
  #     for im in img:
  #       print(im.shape)
  #       plt.imshow(im)
  #       plt.show()