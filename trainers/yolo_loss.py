import numpy as np
import tensorflow as tf
from utils.nms_utils import gpu_nms


def process_output(feature_map, anchors, input_shape, num_classes=80, training=True):
  anchors = tf.reshape(anchors, shape=[1, 1, 1, 3, 2])
  grid_size = feature_map.shape[1:3]  # y,x
  feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + num_classes])
  box_centers, box_wh, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, num_classes], axis=-1)  # xywh

  # get a meshgrid offset
  grid_x = tf.range(grid_size[1], dtype=tf.int32)
  grid_y = tf.range(grid_size[0], dtype=tf.int32)
  grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
  x_offset = tf.reshape(grid_x, (-1, 1))
  y_offset = tf.reshape(grid_y, (-1, 1))
  xy_offset = tf.concat([x_offset, y_offset], axis=-1)
  xy_offset = tf.cast(tf.reshape(xy_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

  # normalize xy according to grid_size,equal to normalize to 416
  box_centers = (tf.nn.sigmoid(box_centers) + xy_offset) / tf.cast(grid_size[::-1], tf.float32)
  # normalize wh according to inputsize 416
  box_wh = tf.exp(box_wh) * anchors / tf.cast(input_shape[::-1], tf.float32)
  box_conf = tf.nn.sigmoid(conf_logits)
  box_prob = tf.nn.sigmoid(prob_logits)
  if training:
    return xy_offset, feature_map, box_centers, box_wh
  else:
    return box_centers, box_wh, box_conf, box_prob


def predict_yolo(feature_map_list, anchors, inputshape, imgshape, padscale):
  anchors = tf.reshape(tf.convert_to_tensor(anchors), (3, 3, 2))
  anchors = tf.cast(anchors, tf.float32)
  boxes = []
  scores = []

  for idx in range(3):
    _feature, _anchor = feature_map_list[idx], anchors[idx]
    _feature = tf.expand_dims(_feature, 0)
    _boxes_center, _boxes_wh, _conf, _classes = process_output(_feature, _anchor, inputshape, training=False)
    _score = tf.reshape(_conf * _classes, [1, -1, 80])

    _boxes_center = _boxes_center / padscale[::-1] * imgshape[::-1]
    _boxes_wh = _boxes_wh / padscale[::-1] * imgshape[::-1]

    _boxes = tf.concat((_boxes_center, _boxes_wh), axis=-1)
    _boxes = tf.reshape(_boxes, shape=[1, -1, 4])
    boxes.append(_boxes)
    scores.append(_score)
  allboxes = tf.concat(boxes, axis=1)
  allscores = tf.concat(scores, axis=1)

  center_x, center_y, width, height = tf.split(allboxes, [1, 1, 1, 1], axis=-1)
  x_min = center_x - width / 2
  y_min = center_y - height / 2
  x_max = center_x + width / 2
  y_max = center_y + height / 2
  allboxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
  nms_boxes, nms_scores, labels = gpu_nms(allboxes, allscores, 80)

  return nms_boxes, nms_scores, labels


def loss_yolo(feature_map_list, gt_list, anchors, inputshape):
  anchors = tf.reshape(tf.convert_to_tensor(anchors), (3, 3, 2))
  anchors = tf.cast(anchors, tf.float32)
  inputshape = tf.cast(inputshape, tf.float32)
  batchsize = tf.cast(feature_map_list[0].shape[0], tf.float32)
  batch_box = 0
  batch_conf = 0
  batch_class = 0

  for idx in range(3):
    _grid_shape = tf.cast(feature_map_list[idx].shape[1:3], tf.float32)
    _featurelist, _anchor, _gt = feature_map_list[idx], anchors[idx], gt_list[idx]
    _object_mask = _gt[..., 4:5]
    _true_class_probs = _gt[..., 5:]

    _xy_offset, _feature, _box_centers, _box_wh = process_output(_featurelist, _anchor, inputshape)

    # get ignoremask
    _valid_true_boxes = tf.boolean_mask(_gt[..., 0:4], tf.cast(_object_mask[..., 0], 'bool'))
    _valid_true_xy = _valid_true_boxes[:, 0:2]
    _valid_true_wh = _valid_true_boxes[:, 2:4]
    _ious = broadcast_iou(true_xy=_valid_true_xy, true_wh=_valid_true_wh,
                          pred_xy=_box_centers, pred_wh=_box_wh)
    _best_iou = tf.reduce_max(_ious, axis=-1)
    _ignore_mask = tf.cast(_best_iou < 0.5, tf.float32)
    _ignore_mask = tf.expand_dims(_ignore_mask, -1)
    # manipulate the gt

    _raw_true_xy = _gt[..., :2] * _grid_shape[::-1] - _xy_offset
    _raw_true_wh = _gt[..., 2:4] / _anchor * inputshape[::-1]
    _raw_true_wh = tf.where(condition=tf.equal(_raw_true_wh,0),x=tf.ones_like(_raw_true_wh),y=_raw_true_wh)
    _raw_true_wh = tf.math.log(_raw_true_wh)
    _box_loss_scale = 2 - _gt[..., 2:3] * _gt[..., 3:4]

    _xy_loss=_object_mask*_box_loss_scale*tf.nn.sigmoid_cross_entropy_with_logits(labels=_raw_true_xy,logits=_feature[...,0:2])
    # _xy_loss = _object_mask * _box_loss_scale * 0.5 * tf.square(_raw_true_xy-tf.nn.sigmoid(_feature[..., 0:2]))
    _wh_loss = _object_mask * _box_loss_scale * 0.5 * tf.square(_raw_true_wh - _feature[..., 2:4])
    _conf_loss = _object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=_object_mask,
                                                                        logits=_feature[..., 4:5]) + \
                 (1 - _object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=_object_mask,
                                                                              logits=_feature[..., 4:5]) * _ignore_mask
    _class_loss = _object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=_true_class_probs,
                                                                         logits=_feature[..., 5:])
    batch_box += tf.reduce_sum(_xy_loss) / batchsize
    batch_box += tf.reduce_sum(_wh_loss) / batchsize
    batch_conf += tf.reduce_sum(_conf_loss) / batchsize
    batch_class += tf.reduce_sum(_class_loss) / batchsize

  return batch_box, batch_conf, batch_class


def broadcast_iou(true_xy, true_wh, pred_xy, pred_wh):
  '''
  maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
  note: here we only care about the size match
  '''
  # shape:
  # true_box_??: [V, 2]
  # pred_box_??: [N, 13, 13, 3, 2]

  # shape: [N, 13, 13, 3, 1, 2]
  pred_box_xy = tf.expand_dims(pred_xy, -2)
  pred_box_wh = tf.expand_dims(pred_wh, -2)

  # shape: [1, V, 2]
  true_box_xy = tf.expand_dims(true_xy, 0)
  true_box_wh = tf.expand_dims(true_wh, 0)

  # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
  intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                              true_box_xy - true_box_wh / 2.)
  intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                              true_box_xy + true_box_wh / 2.)
  intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

  # shape: [N, 13, 13, 3, V]
  intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
  # shape: [N, 13, 13, 3, 1]
  pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
  # shape: [1, V]
  true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

  # [N, 13, 13, 3, V]
  iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
  return iou
