import time
from base.base_trainer import BaseTrainer
import tensorflow as tf
from models.yolo.loss.calc_tensor import loss_fn
from models.yolo.loss.newloss import LossCalculator
import json
from trainers.cocoeval import EvaluatorCOCO
from tensorflow.python.keras import metrics
from .yolo_loss import loss_yolo
class Trainer(BaseTrainer):
  def __init__(self, args,config, model, criterion, optimizer, scheduler):
    self.idx2cat=json.load(open("./trainers/coco_idx2cat.json"))
    self.logger_scalas={}
    self.logger_coco=['mAP','mAp@50','mAP@75','mAP@small','mAP@meduim','mAP@large',
                      'AR@1','AR@10','AR@100','AR@small','AR@medium','AR@large']
    self.logger_pic=[]
    super().__init__(args,config, model, criterion,optimizer, scheduler)
  def _get_loggers(self):
    self.TESTevaluator=EvaluatorCOCO(anchors=self.anchors,
                                     inputsize=(self.configs['model']['net_size'],
                                                self.configs['model']['net_size']),
                                     idx2cate=self.idx2cat,
                                     threshold=self.configs['cls_threshold'],
                                     cateNames=self.configs['model']['labels'])

    self.LossBox =metrics.Mean()
    self.LossConf=metrics.Mean()
    self.LossClass = metrics.Mean()
    self.logger_scalas.update({"lossBox":self.LossBox})
    self.logger_scalas.update({"lossConf": self.LossConf})
    self.logger_scalas.update({"lossClass": self.LossClass})
  def _reset_loggers(self):
    self.TESTevaluator.reset()
    self.LossClass.reset_states()
    self.LossConf.reset_states()
    self.LossBox.reset_states()
  @tf.function
  def train_step(self, imgs, labels):
    with tf.GradientTape() as tape:
      outputs = self.model(imgs, training=True)
      loss_box,loss_conf,loss_class = loss_yolo(outputs,labels,anchors=self.anchors,inputshape=(self.net_size,self.net_size))
      loss=tf.sqrt(tf.reduce_sum(loss_box+loss_conf+loss_class))
    grads = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    self.LossBox.update_state(loss_box)
    self.LossConf.update_state(loss_conf)
    self.LossClass.update_state(loss_class)
    return outputs

  def _valid_epoch(self):
    print("validation start")
    for idx_batch, (imgs, imgpath, scale, ori_shapes, *labels) in enumerate(self.test_dataloader):
      if idx_batch==50:
        break
      grids = self.model(imgs, training=False)
      self.TESTevaluator.append(grids, imgpath, scale, ori_shapes,visualize=True)
    result = self.TESTevaluator.evaluate()
    imgs=self.TESTevaluator.visual_imgs
    print(result)
    for k, v in zip(self.logger_coco, result):
      print("metric {}:{}".format(k,v))
    return result,imgs
  def _train_epoch(self):
    with self.trainwriter.as_default():
      for i, (img,imgpath,scale,ori_shapes,*labels) in enumerate(self.train_dataloader):
        self.global_iter+=1
        if self.global_iter%100==0:
          print(self.global_iter)
          for k,v in self.logger_scalas.items():
            print(k,":",v.result().numpy())

        _=self.train_step(img,labels)

        if self.global_iter%self.log_iter==0:
          for k,v in self.logger_scalas.items():
            tf.summary.scalar(k,v.result(),step=self.global_iter)

          result,imgs=self._valid_epoch()
          for k, v in zip(self.logger_coco, result):
            tf.summary.scalar(k, v, step=self.global_iter)
          for i in range(len(imgs)):
            tf.summary.image("detections_{}".format(i), tf.expand_dims(tf.convert_to_tensor(imgs[i]), 0),
                             step=self.global_iter)


          self._reset_loggers()
    self._save_checkpoint()
if __name__ == '__main__':
  import os

  print(os.getcwd())
