from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from yolo.yolo_loss import predict_yolo
import os
from .Evaluator import Evaluator

class EvaluatorCOCO(Evaluator):
  def __init__(self,anchors,inputsize,cateNames,rootpath,score_thres,iou_thres,idx2cate):
    super().__init__(anchors,inputsize,cateNames,rootpath,score_thres,iou_thres)
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.idx2cat=idx2cate
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
