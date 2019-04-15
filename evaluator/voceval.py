from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from yolo.yolo_loss import predict_yolo
from PIL import Image
from utils.dataset_util import PascalVocXmlParser


class EvaluatorVOC:
  def __init__(self, anchors, inputsize, threshold, idx2cate, cateNames, num_images):
    self.anchors = anchors
    self.inputsize = inputsize
    self.cls_threshold = threshold
    self.idx2cat = idx2cate
    self.cat2idx = {int(v): int(k) for k, v in self.idx2cat.items()}
    self.cateNames = cateNames
    self.visual_imgs = []
    self.rec_pred = [[[] for _ in range(num_images)]
                     for _ in range(len(cateNames))]
    self.reset()
  def reset(self):
    self.coco_imgIds = set([])
    self.visual_imgs = []

  def append(self, grids, imgpath, annpath, padscale, orishape, visualize=False):
    grids = [grid.numpy() for grid in grids]
    padscale = padscale.numpy()
    imgpath = imgpath.numpy()
    annpath = annpath.numpy()
    orishape = orishape.numpy()
    for idx in range(imgpath.shape[0]):
      _imgpath = imgpath[idx].decode('UTF-8')
      _annpath = annpath[idx].decode('UTF-8')
      _grid = [feature[idx] for feature in grids]
      _padscale = padscale[idx]
      _orishape = orishape[idx]
      _boxes, _scores, _labels = predict_yolo(_grid, self.anchors, self.inputsize, _orishape, _padscale, num_classes=20)

      if _boxes is not None:  # do have bboxes
        _boxes, _scores, _labels = _boxes.numpy(), _scores.numpy(), _labels.numpy()
        _boxes = np.concatenate((np.expand_dims(_boxes[:, 1], 1), np.expand_dims(_boxes[:, 0], 1),
                                 np.expand_dims(_boxes[:, 3], 1), np.expand_dims(_boxes[:, 2], 1)), 1)
        for i in range(_boxes.shape[0]):
          self.rec_pred.append({
            "image_name": _imgpath.split('/')[-1],
            "category_id": self.idx2cat[str(_labels[i])],
            "bbox": _boxes,
            "score": float(_scores[i])
          })

        if visualize and len(self.visual_imgs) < 10:
          imPre = np.array(Image.open(_imgpath).convert('RGB'))
          imGT = imPre.copy()
          _, bboxes, labels = PascalVocXmlParser(str(_annpath), self.cateNames).parse()
          boxGT = bboxes
          labelGT = labels
          scoreGT = np.ones(shape=(bboxes.shape[0],))

          # print(_boxes[0])
          visualize_boxes(image=imPre, boxes=_boxes, labels=_labels, probs=_scores, class_labels=self.cateNames)
          visualize_boxes(image=imGT, boxes=np.array(boxGT), labels=np.array(labelGT), probs=np.array(scoreGT),
                          class_labels=self.cateNames)
          whitepad = np.zeros(shape=(imPre.shape[0], 10, 3), dtype=np.uint8)
          imshow = np.concatenate((imGT, whitepad, imPre), axis=1)
          self.visual_imgs.append(imshow)
          # plt.imshow(imshow)
          # plt.show()
          # import os
          # savepath='/home/gwl/PycharmProjects/mine/tf2-yolo3/compare/mine'
          # plt.imsave(os.path.join(savepath,'{}.png'.format(len(self.visual_imgs))),imshow)
      # assert 0

  def evaluate(self):
    pass

