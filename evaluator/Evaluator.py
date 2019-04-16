from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from yolo.yolo_loss import predict_yolo
from PIL import Image

class Evaluator:
  def __init__(self,anchors,inputsize,cateNames,rootpath,score_thres=0.01,iou_thres=0.5,visualize=True):
    self.anchors=anchors
    self.inputsize = inputsize
    self.score_thres=score_thres
    self.iou_thres=iou_thres
    self.cateNames = cateNames
    self.dataset_root=rootpath

    self.visual = visualize
    self.visual_imgs = []
    #show 10 images in tensorboard by default
    self.num_visual=10
    self.build_GT()
  def reset(self):
    pass

  def append(self,grids,imgpath,annpath,padscale,orishape):
    raise NotImplementedError

  def build_GT(self):
    raise NotImplementedError

  def evaluate(self):
    raise NotImplementedError

  def append_visulize(self, imgpath, boxesPre, labelsPre, scoresPre, boxGT, labelGT, savepath=None):
    imPre = np.array(Image.open(imgpath).convert('RGB'))
    imGT = imPre.copy()
    scoreGT = np.ones(shape=(boxGT.shape[0],))
    visualize_boxes(image=imPre, boxes=boxesPre, labels=labelsPre, probs=scoresPre, class_labels=self.cateNames)
    visualize_boxes(image=imGT, boxes=np.array(boxGT), labels=np.array(labelGT), probs=np.array(scoreGT),
                    class_labels=self.cateNames)
    whitepad = np.zeros(shape=(imPre.shape[0], 10, 3), dtype=np.uint8)
    imshow = np.concatenate((imGT, whitepad, imPre), axis=1)
    self.visual_imgs.append(imshow)
    plt.imshow(imshow)
    plt.show()
    if savepath:
      import os
      plt.imsave(os.path.join(savepath, '{}.png'.format(len(self.visual_imgs))), imshow)