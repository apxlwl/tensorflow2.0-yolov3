import os
import glob


def get_filelists(path, prefix, suffix):
  return glob.glob(os.path.join(path, '{}.{}'.format(prefix, suffix)))

# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse



class PascalVocXmlParser(object):
  """Parse annotation for 1-annotation file """

  def __init__(self,annfile,labels):
    self.annfile=annfile
    self.root=self._root_tag(self.annfile)
    self.tree=self._tree(self.annfile)
    self.labels=labels
  def parse(self):
    fname= self.get_fname()
    labels=self.get_labels()
    boxes=self.get_boxes()
    assert os.path.exists(fname),"file {} does not exist".format(fname)
    return fname,np.array(boxes),labels
  def get_fname(self):
    return os.path.join(self.root.find("path").text,
                        self.root.find("filename").text)

  def get_width(self):
    for elem in self.tree.iter():
      if 'width' in elem.tag:
        return int(elem.text)

  def get_height(self):
    for elem in self.tree.iter():
      if 'height' in elem.tag:
        return int(elem.text)

  def get_labels(self):
    labels = []
    obj_tags = self.root.findall("object")
    for t in obj_tags:
      labels.append(self.labels.index(t.find("name").text))
    return labels

  def get_boxes(self):
    bbs = []
    obj_tags = self.root.findall("object")
    for t in obj_tags:
      box_tag = t.find("bndbox")
      x1 = box_tag.find("xmin").text
      y1 = box_tag.find("ymin").text
      x2 = box_tag.find("xmax").text
      y2 = box_tag.find("ymax").text
      box = np.array([int(float(y1)), int(float(x1)), int(float(y2)), int(float(x2))])
      bbs.append(box)
    bbs = np.array(bbs)
    return bbs

  def _root_tag(self, fname):
    tree = parse(fname)
    root = tree.getroot()
    return root

  def _tree(self, fname):
    tree = parse(fname)
    return tree

def _create_empty_grid(net_size, n_classes, n_boxes=3,downsample_ratio=32):
  base_grid_h, base_grid_w = net_size // downsample_ratio, net_size // downsample_ratio
  ys_1 = np.zeros((1 * base_grid_h, 1 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 1
  ys_2 = np.zeros((2 * base_grid_h, 2 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 2
  ys_3 = np.zeros((4 * base_grid_h, 4 * base_grid_w, n_boxes, 4 + 1 + n_classes))  # desired network output 3
  list_ys = [ys_3, ys_2, ys_1]
  return list_ys


def _encode_box(grid, original_box, anchor_box, net_w, net_h):
  y1, x1, y2, x2 = original_box
  _, _, anchor_w, anchor_h = anchor_box
  grid_h, grid_w = grid.shape[:2]
  center_x = .5 * (x1 + x2)
  center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
  center_y = .5 * (y1 + y2)
  center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

  # determine the sizes of the bounding box
  w = np.log(max((x2 - x1), 1) / float(anchor_w))  # t_w
  h = np.log(max((y2 - y1), 1) / float(anchor_h))  # t_h

  box = [center_x, center_y, w, h]
  return box


def _assign_box(yolo, box_index, box, label):
  center_x, center_y, _, _ = box

  # determine the location of the cell responsible for this object
  grid_x = int(np.floor(center_x))
  grid_y = int(np.floor(center_y))

  # assign ground truth x, y, w, h, confidence and class probs to y_batch
  yolo[grid_y, grid_x, box_index] = 0.0
  yolo[grid_y, grid_x, box_index, 0:4] = box
  yolo[grid_y, grid_x, box_index, 4] = 1.0
  yolo[grid_y, grid_x, box_index, 5 + label] = 1.0

class DataGenerator:
  def __init__(self, dataset, shuffle=False):
    self.dataset = dataset
    self.shuffle = shuffle

  def __call__(self):
    indices = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.shuffle(indices)
    for img_idx in indices:
      img,ori_img, imgmeta,label0,label1,label2 = self.dataset[img_idx]
      yield img,ori_img,imgmeta,label0,label1,label2
if __name__ == '__main__':
  print(get_filelists('./svhn/train/anns',prefix='*',suffix='xml'))
