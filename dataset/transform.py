import numpy as np

import dataset.image as timage
import dataset.bbox as tbbox

class YOLO3DefaultValTransform(object):
  """Default YOLO validation transform.

  Parameters
  ----------
  width : int
      Image width.
  height : int
      Image height.
  mean : array-like of size 3
      Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
  std : array-like of size 3
      Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

  """

  def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    self._width = width
    self._height = height
    self._mean = mean
    self._std = std

  def __call__(self, img, bbox):
    """Apply transform to validation image/label."""
    # resize
    h, w, _ = img.shape
    img = timage.img_resize(img, out_size=(self._width, self._height))
    bbox = tbbox.bbox_resize(bbox, (w, h), (self._width, self._height))
    img=timage.imnormalize(img,self._mean,self._std)
    return img, bbox.astype(img.dtype)


class YOLO3DefaultTrainTransform(object):
  def __init__(self, width, height, mean=(0.485, 0.456, 0.406),
               std=(0.229, 0.224, 0.225)):
    self._width = width
    self._height = height
    self._mean = mean
    self._std = std

  def __call__(self, img, bbox):
    """
    :param image:np.array HWC
    :param bbox:np.array box N,4 x1y1x2y2
    :return:
    """
    img=timage.random_color_distort(img)
    # random expansion with prob 0.5
    if np.random.uniform(0, 1) > 0:
      img, expand = timage.random_expand(img)
      bbox = tbbox.translate(bbox, x_offset=expand[0], y_offset=expand[1])
    else:
      img, bbox = img, bbox

    # random cropping
    h, w, _ = img.shape
    bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
    x0, y0, w, h = crop
    img=timage.fixed_crop(img,x0,y0,w,h)

    #resize
    img=timage.img_resize(img,out_size=(self._width,self._height))
    bbox = tbbox.bbox_resize(bbox,(w,h),(self._width,self._height))

    #flip
    h, w, _ = img.shape
    img,flips=timage.random_flip(img,px=1)
    bbox = tbbox.bbox_flip(bbox,(w,h),flip_x=flips[0])

    #normalize
    img=timage.imnormalize(img,self._mean,self._std)

    return img,bbox

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

if __name__ == '__main__':
  from PIL import Image
  import matplotlib.pyplot as plt
  import os
  from utils.visualize import visualize_boxes
  train_transform=YOLO3DefaultValTransform(width=416,height=416)
  root = '/disk2/datasets/coco/images/val2017/'
  # root = '/home/gwl/datasets/coco2017/images/val2017/'
  # filelist = os.listdir(root)
  imagename = '000000532481.jpg'
  bboxes = [[250.82, 168.26, 320.93, 233.14],
            [435.35, 294.23, 448.81, 302.04],
            [447.44, 293.91, 459.6, 301.56],
            [460.59, 291.71, 473.34, 300.16],
            [407.07, 287.25, 419.72, 297.11],
            [618.06, 289.31, 629.66, 297.26],
            [512.3, 294.07, 533.48, 299.64],
            [285.55, 370.56, 297.62, 389.77],
            [61.61, 43.76, 107.89, 122.05],
            [238.54, 158.48, 299.7, 213.87]]
  names = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
  ]
  bboxes=np.array(bboxes)

  # img = Image.open(os.path.join(root, imagename))
  # plt.imshow(np.array(img))
  # plt.show()
  # print(img.size)
  # img=img.crop((20,100,300,300))
  # plt.imshow(np.array(img))
  # plt.show()
  # print(img.size)

  img = np.array(Image.open(os.path.join(root, imagename)))
  img,bboxes = train_transform(img,bboxes)
  #x1y1x2y2->y1x1y2x2
  bboxes = np.concatenate((np.expand_dims(bboxes[:,1],1),
                           np.expand_dims(bboxes[:,0],1),
                           np.expand_dims(bboxes[:,3],1),
                           np.expand_dims(bboxes[:,2],1),
                           ),axis=1)

  visualize_boxes(img, boxes=np.array(bboxes),
                  labels=np.array([0, 2, 2, 2, 2, 2, 2, 0, 33, 37]),
                  probs=np.ones(shape=(10,), dtype=np.float32), class_labels=names)
  plt.imshow(img/255)
  plt.show()
