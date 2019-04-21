from base.base_trainer import BaseTrainer
import tensorflow as tf
from evaluator.cocoeval import EvaluatorCOCO

class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer):
    super().__init__(args, model, optimizer)

  def _get_loggers(self):
    #get public loggers
    super()._get_loggers()
    self.TESTevaluator = EvaluatorCOCO(anchors=self.anchors,
                                       cateNames=self.labels,
                                       rootpath=self.dataset_root,
                                       score_thres=0.01,
                                       iou_thres=0.5,
                                       )
    # get customized loggers
    self.logger_custom = ['mAP', 'mAp@50', 'mAP@75', 'mAP@small', 'mAP@meduim', 'mAP@large',
                        'AR@1', 'AR@10', 'AR@100', 'AR@small', 'AR@medium', 'AR@large']


  def _valid_epoch(self,multiscale=False,flip=False):
    results, imgs = super()._valid_epoch(multiscale=multiscale,flip=flip)
    return results, imgs

  def _train_epoch(self):
    with self.summarywriter.as_default():
      for i, (img, imgpath,annpath, scale, ori_shapes, *labels) in enumerate(self.train_dataloader):
        self.global_iter.assign_add(1)

        self.train_step(img, labels)

        if self.global_iter.numpy() % self.log_iter == 0:
          for k, v in self.logger_losses.items():
            tf.summary.scalar(k, v.result(), step=self.global_iter.numpy())
          result, imgs = self._valid_epoch()
          for k, v in zip(self.logger_coco, result):
            tf.summary.scalar(k, v, step=self.global_iter.numpy())
          for i in range(len(imgs)):
            tf.summary.image("detections_{}".format(i), tf.expand_dims(tf.convert_to_tensor(imgs[i]), 0),
                             step=self.global_iter.numpy())
          self._reset_loggers()
    self.ckpt_manager.save(self.global_epoch)
