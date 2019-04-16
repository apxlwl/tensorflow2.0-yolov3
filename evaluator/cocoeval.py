from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from yolo.yolo_loss import predict_yolo
import os
from .Evaluator import Evaluator

class EvaluatorCOCO(Evaluator):
  def __init__(self,anchors,inputsize,cateNames,rootpath,score_thres,iou_thres):
    super().__init__(anchors,inputsize,cateNames,rootpath,score_thres,iou_thres)
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.idx2cat= {
                 "0": 1,
                 "1": 2,
                 "2": 3,
                 "3": 4,
                 "4": 5,
                 "5": 6,
                 "6": 7,
                 "7": 8,
                 "8": 9,
                 "9": 10,
                 "10": 11,
                 "11": 13,
                 "12": 14,
                 "13": 15,
                 "14": 16,
                 "15": 17,
                 "16": 18,
                 "17": 19,
                 "18": 20,
                 "19": 21,
                 "20": 22,
                 "21": 23,
                 "22": 24,
                 "23": 25,
                 "24": 27,
                 "25": 28,
                 "26": 31,
                 "27": 32,
                 "28": 33,
                 "29": 34,
                 "30": 35,
                 "31": 36,
                 "32": 37,
                 "33": 38,
                 "34": 39,
                 "35": 40,
                 "36": 41,
                 "37": 42,
                 "38": 43,
                 "39": 44,
                 "40": 46,
                 "41": 47,
                 "42": 48,
                 "43": 49,
                 "44": 50,
                 "45": 51,
                 "46": 52,
                 "47": 53,
                 "48": 54,
                 "49": 55,
                 "50": 56,
                 "51": 57,
                 "52": 58,
                 "53": 59,
                 "54": 60,
                 "55": 61,
                 "56": 62,
                 "57": 63,
                 "58": 64,
                 "59": 65,
                 "60": 67,
                 "61": 70,
                 "62": 72,
                 "63": 73,
                 "64": 74,
                 "65": 75,
                 "66": 76,
                 "67": 77,
                 "68": 78,
                 "69": 79,
                 "70": 80,
                 "71": 81,
                 "72": 82,
                 "73": 84,
                 "74": 85,
                 "75": 86,
                 "76": 87,
                 "77": 88,
                 "78": 89,
                 "79": 90
               }
    self.cat2idx= {int(v): int(k) for k, v in self.idx2cat.items()}
    self.reset()

  def reset(self):
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.visual_imgs=[]
  def build_GT(self):
    self.cocoGt = COCO(os.path.join(self.dataset_root,'/annotations/instances_val2017.json'))

  def append(self,grids,imgpath,annpath,padscale,orishape):
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
      _boxes,_scores,_labels = predict_yolo(_grid, self.anchors, self.inputsize, _orishape,_padscale,num_classes=80)
      if _boxes is not None: #do have bboxes
        _boxes,_scores,_labels = _boxes.numpy(),_scores.numpy(),_labels.numpy()
        for i in range(_boxes.shape[0]):
          self.coco_imgIds.add(_image_id)
          self.coco_results.append({
            "image_id": _image_id,
            "category_id": self.idx2cat[str(_labels[i])],
            "bbox": [_boxes[i][1], _boxes[i][0], _boxes[i][3] - _boxes[i][1], _boxes[i][2] - _boxes[i][0]],
            "score": float(_scores[i])
          })
        if len(self.visual_imgs)<self.num_visual:
          annIDs=self.cocoGt.getAnnIds(imgIds=[_image_id])
          boxGT=[]
          labelGT=[]
          for id in annIDs:
            ann=self.cocoGt.anns[id]
            x,y,w,h=ann['bbox']
            boxGT.append([x,y,x+w,y+h])
            labelGT.append(self.cat2idx[ann['category_id']])
          self.append_visulize(_imgpath,_boxes,_labels,_scores,boxGT,labelGT)
  def evaluate(self):
    try:
      cocoDt = self.cocoGt.loadRes(self.coco_results)
    except:
      print("no boxes detected, coco eval aborted")
      return 1
    cocoEval = COCOeval(self.cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = list(self.coco_imgIds)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats
