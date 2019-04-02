from datasets.pycocotools.coco import COCO
from datasets.pycocotools.cocoeval import COCOeval
import json
from models.yolo.post_proc.decoder import postprocess_ouput
from models.yolo.utils.box import boxes_to_array,to_minmax
from utils.visualize import visualize_boxes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class EvaluatorCOCO:
  def __init__(self,anchors,inputsize,threshold,idx2cate,cateNames):
    self.anchors=anchors
    self.inputsize=inputsize
    self.cls_threshold=threshold
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.idx2cat=idx2cate
    self.cateNames=cateNames
    self.reset()
    self.visual_imgs=[]
  def reset(self):
    self.coco_imgIds=set([])
    self.coco_results=[]
  def append(self,grids,imgpath,padscale,orishape,visualize=False):
    grids = [grid.numpy() for grid in grids]
    padscale = padscale.numpy()
    imgpath = imgpath.numpy()
    orishape = orishape.numpy()
    for idx in range(imgpath.shape[0]):
      _imgpath = imgpath[idx].decode('UTF-8')
      _image_id = int(_imgpath[-16:-4])
      _grid = [feature[idx] for feature in grids]
      _padscale = padscale[idx]
      _orishape=orishape[idx]
      _classboxes = postprocess_ouput(_grid, self.anchors, self.inputsize, _orishape,_padscale)
      if len(_classboxes) > 0:
        _boxes, _probs = boxes_to_array(_classboxes)
        _boxes = to_minmax(_boxes)
        _labels = np.array([b.get_label() for b in _classboxes])
        _boxes = _boxes[_probs >= self.cls_threshold]
        _labels = _labels[_probs >= self.cls_threshold]
        _probs = _probs[_probs >= self.cls_threshold]
      else:
        _boxes, _labels, _probs = [], [], []
      for i in range(len(_boxes)):
        self.coco_imgIds.add(_image_id)
        self.coco_results.append({
          "image_id": _image_id,
          "category_id": self.idx2cat[str(_labels[i])],
          "bbox": [_boxes[i][0], _boxes[i][1], _boxes[i][2] - _boxes[i][0], _boxes[i][3] - _boxes[i][1]],
          "score": float(_probs[i])
        })
      if visualize and len(self.visual_imgs)<10:
        imshow = np.array(plt.imread(_imgpath))
        visualize_boxes(image=imshow, boxes=_boxes, labels=_labels, probs=_probs, class_labels=self.cateNames)
        self.visual_imgs.append(imshow)


  def evaluate(self):
    cocoGt = COCO('/home/gwl/datasets/coco2017/annotations/instances_val2017.json')
    cocoDt = cocoGt.loadRes(self.coco_results)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = list(self.coco_imgIds)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats