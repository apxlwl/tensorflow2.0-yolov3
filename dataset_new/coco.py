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
    self.network_size = scale[0]
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
    self.anchors = np.array(configs['anchors'])
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
      # bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
      bbox = [x1,y1,x1+w,y1+h]
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
    bboxes = ann['bboxes'] #[y1,x1,y2,x2]
    labels = ann['labels']
    print(osp.join(self.image_dir, img_info['file_name']))
    print(bboxes)
    print(labels)
    assert 0
    flip = True if np.random.rand() < self.flip_ratio else False
    flip = False
    # Handle the image
    img, img_shape, scale_factor = self.img_transform(img, flip)
    pad_shape = img.shape[:2]
    pad_scale = np.array(ori_shape) * scale_factor / np.array(pad_shape[:2])
    # Handle the annotation.
    bboxes, labels = self.bbox_transform(bboxes, labels, img_shape, scale_factor, flip)
    #y1,x1,y2,x2->x1,y1,x2,y2
    bboxes=np.stack((bboxes[:,1],bboxes[:,0],bboxes[:,3],bboxes[:,2]),axis=1)
    list_grids = preprocess(bboxes, labels, img.shape[:2], class_num=80, anchors=self.anchors)

    return img.astype(np.float32), \
           osp.join(self.image_dir, img_info['file_name']), \
           np.array(pad_scale).astype(np.float32), \
           np.array(ori_shape).astype(np.float32), \
           list_grids[0].astype(np.float32), \
           list_grids[1].astype(np.float32), \
           list_grids[2].astype(np.float32),


def preprocess(boxes,labels,input_shape,class_num,anchors):
  '''
  :param boxes:n,x,y,x2,y2
  :param labels: n,1
  :param img_size:(h,w)
  :param class_num:
  :param anchors:(9,2)
  :return:
  '''
  input_shape=np.array(input_shape)
  #find match anchor for each box,leveraging numpy broadcasting tricks
  boxes_center=(boxes[...,2:4]+boxes[...,0:2])//2
  boxes_wh=boxes[...,2:4]-boxes[...,0:2]
  boxes_wh=np.expand_dims(boxes_wh,1)
  min_wh=np.maximum(-boxes_wh/2,-anchors/2)
  max_wh=np.minimum(boxes_wh/2,anchors/2)
  intersect_wh=max_wh-min_wh
  intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
  box_area=boxes_wh[...,0]*boxes_wh[...,1]
  anchors_area=anchors[...,0]*anchors[...,1]
  iou=intersect_area/(box_area+anchors_area-intersect_area)
  best_ious=np.argmax(iou,axis=1)
  #normalize boxes according to inputsize(416)
  boxes[...,0:2]=boxes_center/input_shape[::-1]
  boxes[...,2:4]=np.squeeze(boxes_wh,1)/input_shape[::-1]
  #get dummy gt with zeros
  y_true_52 = np.zeros((input_shape[1] // 8, input_shape[0] // 8, 3, 5 + class_num), np.float32)
  y_true_26 = np.zeros((input_shape[1] // 16, input_shape[0] // 16, 3, 5 + class_num), np.float32)
  y_true_13 = np.zeros((input_shape[1] // 32, input_shape[0] // 32, 3, 5 + class_num), np.float32)
  y_true_list=[y_true_52,y_true_26,y_true_13]
  grid_shapes=[input_shape//8,input_shape//16,input_shape//32]

  for idx,match_id in enumerate(best_ious):
    group_idx=match_id//3
    sub_idx=match_id%3
    idx_x=np.floor(boxes[idx,0]*grid_shapes[group_idx]).astype('int32')
    idx_y=np.floor(boxes[idx,1]*grid_shapes[group_idx]).astype('int32')

    y_true_list[group_idx][idx_y,idx_x,sub_idx,:2]=boxes[idx,0:2]
    y_true_list[group_idx][idx_y,idx_x,sub_idx,2:4]=boxes[idx,2:4]
    y_true_list[group_idx][idx_y,idx_x,sub_idx,4]=1.
    y_true_list[group_idx][idx_y,idx_x,sub_idx,5+labels[idx]]=1.
  return y_true_list

class DataGenerator:
  def __init__(self, dataset, shuffle=False):
    self.dataset = dataset
    self.shuffle = shuffle

  def __call__(self):
    indices = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.shuffle(indices)
    for img_idx in indices:
      img, imgpath, scale, ori_shape, label0, label1, label2 = self.dataset[img_idx]
      yield img, imgpath, scale, ori_shape, label0, label1, label2


def get_dataset(config):
  config["subset"] = 'val'
  valset = CocoDataSet(config)
  generator = DataGenerator(valset)
  valset = tf.data.Dataset.from_generator(generator,
                                          ((tf.float32, tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                                            tf.float32)))
  valset = valset.batch(config['batch_size'])
  return valset,valset
  config["subset"] = 'train'
  trainset = CocoDataSet(config)
  generator = DataGenerator(trainset)
  trainset = tf.data.Dataset.from_generator(generator,
                                            ((tf.float32, tf.string, tf.float32, tf.float32, tf.float32, tf.float32,
                                              tf.float32)))
  trainset = trainset.batch(config['batch_size'])

  return trainset, valset


if __name__ == '__main__':
  import json
  import matplotlib.pyplot as plt

  with open('../configs/coco.json', 'r') as f:
    configs = json.load(f)
  configs['dataset']['dataset_dir']='/disk2/datasets/coco'
  train, _ = get_dataset(configs['dataset'])
  for i, inputs in enumerate(train):
    img = inputs[0][0].numpy()
    plt.imshow(img / img.max())
    plt.show()
    assert 0
