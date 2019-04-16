from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from yolo.yolo_loss import predict_yolo
from PIL import Image

class Evaluator:
  def __init__(self,anchors,inputsize,score_thres=0.01,iou_thres=0.5,visualize=True):
    self.anchors=anchors
    self.inputsize = inputsize
    self.score_thres=score_thres
    self.iou_thres=iou_thres
    self.visual = visualize
    self.visual_imgs = []
  def reset(self):
    pass

  def append(self,grids,imgpath,padscale,orishape):
    '''
    evaluate and append results for single image
    :param grids:
    :param imgpath:
    :param padscale:
    :param orishape:
    :return:
    '''
    raise

  def build_GT(self):
    pass

  def evaluate(self):
    pass