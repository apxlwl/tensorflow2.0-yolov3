import numpy as np

import datasets.image as timage
import datasets.bbox as tbbox


class ImageTransform(object):
  '''Preprocess the image.

      1. rescale the image to expected size
      2. normalize the image
      3. flip the image (if needed)
      4. pad the image (if needed)
  '''

  def __init__(self,
               scale=(800, 1333),
               mean=(0, 0, 0),
               std=(1, 1, 1),
               pad_mode='fixed'):
    self.scale = scale
    self.mean = mean
    self.std = std
    self.pad_mode = pad_mode

    self.impad_size = max(scale) if pad_mode == 'fixed' else 64

  def __call__(self, img, flip=False):
    img = random_color_distort(img)
    img, scale_factor = imrescale(img, self.scale)
    img_shape = img.shape
    img = imnormalize(img, self.mean, self.std)

    if flip:
      img = img_flip(img)
    if self.pad_mode == 'fixed':
      img = impad_to_square(img, self.impad_size)

    else:  # 'non-fixed'
      img = impad_to_multiple(img, self.impad_size)

    return img, img_shape, scale_factor


class BboxTransform(object):
  '''Preprocess ground truth bboxes.

      1. rescale bboxes according to image size
      2. flip bboxes (if needed)
  '''

  def __init__(self):
    pass

  def __call__(self, bboxes, labels,
               img_shape, scale_factor, flip=False):
    bboxes = bboxes * scale_factor
    if flip:
      bboxes = bbox_flip(bboxes, img_shape)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[0])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[1])

    return bboxes, labels


class YOLO3TrainTransform(object):
  def __init__(self, width, height, mean=(0, 0, 0), std=(1, 1, 1)):
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
    print(crop)
    assert 0
    img = mx.image.fixed_crop(img, x0, y0, w, h)
    return img,bbox

if __name__ == '__main__':
  from PIL import Image
  import matplotlib.pyplot as plt
  import os
  from utils.visualize import visualize_boxes
  train_transform=YOLO3TrainTransform(width=416,height=416)
  root = '/disk2/datasets/coco/images/val2017/'
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
  plt.imshow(img / img.max())
  plt.show()
