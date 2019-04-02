import os
import os.path as osp
import cv2
import numpy as np
from pycocotools.coco import COCO
from datasets import transform
from utils.dataset_util import _create_empty_grid, _assign_box, _encode_box
from utils.box_util import find_match_anchor, create_anchor_boxes
from datasets import COCO_ANCHORS
import tensorflow as tf

tf.config.gpu.set_per_process_memory_growth(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DOWNSAMPLE_RATIO = 32
DEFAULT_NETWORK_SIZE = 416


class CocoDataSet(object):
  def __init__(self, configs):
    '''Load a subset of the COCO dataset.

    Attributes
    ---
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val).
        flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
        pad_mode: Which padded method to use (fixed, non-fixed)
        mean: Tuple. Image mean.
        std: Tuple. Image standard deviation.
        scale: Tuple of two integers.
    '''
    mean = (0, 0, 0)
    std = (1, 1, 1)
    debug = configs["debug"]
    pad_mode = configs["pad_mode"]
    dataset_dir = configs["dataset_dir"]
    subset = configs["subset"]
    scale = (configs["width"], configs["height"])
    self.network_size=scale[0]
    self.flip_ratio = 0 if subset == 'val' else configs["flip"]
    if subset not in ['train', 'val']:
      raise AssertionError('subset must be "train" or "val".')

    self.coco = COCO("{}/annotations/instances_{}2017.json".format(dataset_dir, subset))

    # get the mapping from original category ids to labels
    self.cat_ids = self.coco.getCatIds()
    self.cat2label = {
      cat_id: i
      for i, cat_id in enumerate(self.cat_ids)
    }
    self.img_ids, self.img_infos = self._filter_imgs()

    self.image_dir = "{}/images/{}2017".format(dataset_dir, subset)

    if pad_mode in ['fixed', 'non-fixed']:
      self.pad_mode = pad_mode
    elif subset == 'train':
      self.pad_mode = 'fixed'
    else:
      self.pad_mode = 'non-fixed'
    self.anchors = create_anchor_boxes(COCO_ANCHORS)
    self.img_transform = transform.ImageTransform(scale, mean, std, pad_mode)
    self.bbox_transform = transform.BboxTransform()

  def _filter_imgs(self, min_size=32):
    '''Filter images too small or without ground truths.

    Args
    ---
        min_size: the minimal size of the image.
    '''
    # Filter images without ground truths.
    all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
    # Filter images too small.
    img_ids = []
    img_infos = []
    for i in all_img_ids:
      info = self.coco.loadImgs(i)[0]
      ann_ids = self.coco.getAnnIds(imgIds=i)
      ann_info = self.coco.loadAnns(ann_ids)
      ann = self._parse_ann_info(ann_info)
      if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
        img_ids.append(i)
        img_infos.append(info)
    return img_ids, img_infos

  def _load_ann_info(self, idx):
    img_id = self.img_ids[idx]
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    ann_info = self.coco.loadAnns(ann_ids)
    return ann_info

  def _parse_ann_info(self, ann_info):
    '''Parse bbox annotation.

    Args
    ---
        ann_info (list[dict]): Annotation info of an image.

    Returns
    ---
        dict: A dict containing the following keys: bboxes,
            bboxes_ignore, labels.
    '''
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []

    for i, ann in enumerate(ann_info):
      if ann.get('ignore', False):
        continue
      x1, y1, w, h = ann['bbox']
      if ann['area'] <= 0 or w < 1 or h < 1:
        continue
      bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
      if ann['iscrowd']:
        gt_bboxes_ignore.append(bbox)
      else:
        gt_bboxes.append(bbox)
        gt_labels.append(self.cat2label[ann['category_id']])

    if gt_bboxes:
      gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
      gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
      gt_bboxes = np.zeros((0, 4), dtype=np.float32)
      gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
      gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
      gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
      bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

    return ann

  def __len__(self):
    return len(self.img_infos)

  def __getitem__(self, idx):
    '''Load the image and its bboxes for the given index.

    Args
    ---
        idx: the index of images.

    Returns
    ---
        tuple: A tuple containing the following items: image,
            bboxes, labels.
    '''
    img_info = self.img_infos[idx]
    ann_info = self._load_ann_info(idx)

    # load the image.
    assert os.path.exists(osp.join(self.image_dir, img_info['file_name']))
    img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_shape = img.shape[:2]

    # Load the annotation.
    ann = self._parse_ann_info(ann_info)
    bboxes = ann['bboxes']
    labels = ann['labels']

    flip = True if np.random.rand() < self.flip_ratio else False

    # Handle the image
    img, img_shape, scale_factor = self.img_transform(img, flip)
    pad_shape = img.shape[:2]
    pad_scale=np.array(ori_shape)*scale_factor/np.array(pad_shape[:2])
    # Handle the annotation.
    bboxes, labels = self.bbox_transform(
      bboxes, labels, img_shape, scale_factor, flip)
    list_grids = _create_empty_grid(self.network_size, n_classes=80)

    for box, label in zip(bboxes, labels):
      match_anchor, scale_index, box_index = find_match_anchor(box, self.anchors)
      coded_box = _encode_box(list_grids[scale_index], box, match_anchor, self.network_size, self.network_size)
      _assign_box(list_grids[scale_index], box_index, coded_box, label)
    return img.astype(np.float32), \
           osp.join(self.image_dir, img_info['file_name']), \
           np.array(pad_scale).astype(np.float32), \
           np.array(ori_shape).astype(np.float32), \
           list_grids[0].astype(np.float32), \
           list_grids[1].astype(np.float32), \
           list_grids[2].astype(np.float32),

  def get_categories(self):
    return ['bg'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]

class DataGenerator:
  def __init__(self, dataset, shuffle=False):
    self.dataset = dataset
    self.shuffle = shuffle

  def __call__(self):
    indices = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.shuffle(indices)
    for img_idx in indices:
      img,imgpath, scale,ori_shape,label0,label1,label2 = self.dataset[img_idx]
      yield img,imgpath,scale,ori_shape,label0,label1,label2

def get_dataset(config):
  config["subset"] = 'val'
  valset = CocoDataSet(config)
  generator = DataGenerator(valset)
  valset = tf.data.Dataset.from_generator(generator,
                                          ((tf.float32, tf.string,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)))
  valset = valset.batch(config['batch_size'])

  config["subset"] = 'train'
  trainset = CocoDataSet(config)
  generator = DataGenerator(trainset)
  trainset = tf.data.Dataset.from_generator(generator,
                                            ((tf.float32, tf.string,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)))
  trainset = trainset.batch(config['batch_size'])

  return trainset, valset


if __name__ == '__main__':
  import json
  from utils.visualize import draw_boxes
  import matplotlib.pyplot as plt

  with open('../configs/coco.json', 'r') as f:
    configs = json.load(f)
  train, _ = get_dataset(configs['dataset'])
  for i, inputs in enumerate(train):
    img = inputs[0][0].numpy()
    plt.imshow(img / img.max())
    plt.show()
    assert 0
