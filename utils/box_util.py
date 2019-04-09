import numpy as np
import cv2





def create_anchor_boxes(anchors):
  """
  # Args
      anchors : list of floats
  # Returns
      boxes : array, shape of (len(anchors)/2, 4)
          centroid-type
  """
  boxes = []
  n_boxes = int(len(anchors) / 2)
  for i in range(n_boxes):
    boxes.append(np.array([0, 0, anchors[2 * i], anchors[2 * i + 1]]))
  return np.array(boxes)


def to_minmax(centroid_boxes):
  centroid_boxes = centroid_boxes.astype(np.float)
  minmax_boxes = np.zeros_like(centroid_boxes)

  cx = centroid_boxes[:, 0]
  cy = centroid_boxes[:, 1]
  w = centroid_boxes[:, 2]
  h = centroid_boxes[:, 3]

  minmax_boxes[:, 0] = cx - w / 2
  minmax_boxes[:, 1] = cy - h / 2
  minmax_boxes[:, 2] = cx + w / 2
  minmax_boxes[:, 3] = cy + h / 2
  return minmax_boxes


def centroid_box_iou(box1, box2):
  def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
      if x4 < x1:
        return 0
      else:
        return min(x2, x4) - x1
    else:
      if x2 < x3:
        return 0
      else:
        return min(x2, x4) - x3

  _, _, w1, h1 = box1.reshape(-1, )
  _, _, w2, h2 = box2.reshape(-1, )
  x1_min, y1_min, x1_max, y1_max = to_minmax(box1.reshape(-1, 4)).reshape(-1, )
  x2_min, y2_min, x2_max, y2_max = to_minmax(box2.reshape(-1, 4)).reshape(-1, )
  intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
  intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
  intersect = intersect_w * intersect_h
  union = w1 * h1 + w2 * h2 - intersect

  return float(intersect) / union


def find_match_box(centroid_box, anchors):
  """Find the index of the boxes with the largest overlap among the N-boxes.

  # Args
      box : array, shape of (1, 4)
      boxes : array, shape of (N, 4)

  # Return
      match_index : int
  """
  match_index = -1
  max_iou = -1

  for i, anchor in enumerate(anchors):
    iou = centroid_box_iou(centroid_box, anchor)
    if iou > max_iou:
      match_index = i
      max_iou = iou
  return match_index


def find_match_anchor(box, anchor_boxes):
  """
  # Args
      box : array, shape of (4,)
      anchor_boxes : array, shape of (9, 4)
  """
  y1, x1, y2, x2 = box
  shifted_box = np.array([0, 0, x2 - x1, y2 - y1])
  max_index = find_match_box(shifted_box, anchor_boxes)
  max_anchor = anchor_boxes[max_index]
  scale_index = max_index // 3
  box_index = max_index % 3
  return max_anchor, scale_index, box_index


if __name__ == '__main__':
   box1=np.array([412.8, 157.61, 53.05, 138.01])
   box2=np.array([430.0, 164.0, 38.0, 132.0])
   for box in [box1,box2]:
     box[0]+=box[2]/2
     box[1]+=box[3]/2
   print(centroid_box_iou(box2,box1))
