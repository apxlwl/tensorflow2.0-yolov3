from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from yolo.yolo_loss import predict_yolo
from PIL import Image
from .Evaluator import Evaluator
class EvaluatorCOCO:
  def __init__(self,anchors,inputsize,threshold,idx2cate,cateNames):
    self.anchors=anchors
    self.inputsize=inputsize
    self.cls_threshold=threshold
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.idx2cat=idx2cate
    self.cat2idx= {int(v): int(k) for k, v in self.idx2cat.items()}
    self.cateNames=cateNames
    self.reset()
    self.visual_imgs=[]
    self.cocoGt = COCO('/home/gwl/datasets/coco2017/annotations/instances_val2017.json')
  def reset(self):
    self.coco_imgIds=set([])
    self.coco_results=[]
    self.visual_imgs=[]
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
        if visualize and len(self.visual_imgs)<10:
          imPre = np.array(Image.open(_imgpath).convert('RGB'))
          imGT=imPre.copy()
          annIDs=self.cocoGt.getAnnIds(imgIds=[_image_id])
          boxGT=[]
          labelGT=[]
          scoreGT=[]
          for id in annIDs:
            ann=self.cocoGt.anns[id]
            x,y,w,h=ann['bbox']
            boxGT.append([x,y,x+w,y+h])
            labelGT.append(self.cat2idx[ann['category_id']])
            scoreGT.append(1.0)
          _boxes=np.concatenate((np.expand_dims(_boxes[:,1],1),np.expand_dims(_boxes[:,0],1),np.expand_dims(_boxes[:,3],1),np.expand_dims(_boxes[:,2],1)),1)
          visualize_boxes(image=imPre, boxes=_boxes, labels=_labels, probs=_scores, class_labels=self.cateNames)
          visualize_boxes(image=imGT, boxes=np.array(boxGT), labels=np.array(labelGT), probs=np.array(scoreGT), class_labels=self.cateNames)
          whitepad=np.zeros(shape=(imPre.shape[0],10,3),dtype=np.uint8)
          imshow=np.concatenate((imGT,whitepad,imPre),axis=1)
          self.visual_imgs.append(imshow)
          # import os
          # savepath='/home/gwl/PycharmProjects/mine/tf2-yolo3/compare/mine'
          # plt.imsave(os.path.join(savepath,'{}.png'.format(_image_id)),imshow)
          plt.imshow(imshow)
          plt.show()
          assert 0
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
