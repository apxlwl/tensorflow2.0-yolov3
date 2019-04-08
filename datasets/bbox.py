import cv2
import numpy as np
import random


def bbox_flip(bboxes, img_shape):
  '''Flip bboxes horizontally.

  Args
  ---
      bboxes: [..., 4]
      img_shape: Tuple. (height, width)

  Returns
  ---
      np.ndarray: the flipped bboxes.
  '''
  w = img_shape[1]
  flipped = bboxes.copy()
  flipped[..., 1] = w - bboxes[..., 3] - 1
  flipped[..., 3] = w - bboxes[..., 1] - 1
  return flipped

def translate(bbox, x_offset=0, y_offset=0):
  """Translate bounding boxes by offsets.

  Parameters
  ----------
  bbox : numpy.ndarray
      Numpy.ndarray with shape (N, 4+) where N is the number of bounding boxes.
      The second axis represents attributes of the bounding box.
      Specifically, these are :math:`(x_{min}, y_{min}, x_{max}, y_{max})`,
      we allow additional attributes other than coordinates, which stay intact
      during bounding box transformations.
  x_offset : int or float
      Offset along x axis.
  y_offset : int or float
      Offset along y axis.

  Returns
  -------
  numpy.ndarray
      Translated bounding boxes with original shape.
  """
  bbox = bbox.copy()
  bbox[:, :2] += (x_offset, y_offset)
  bbox[:, 2:4] += (x_offset, y_offset)
  return bbox
