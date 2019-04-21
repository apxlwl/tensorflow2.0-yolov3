from base.base_trainer import BaseTrainer
import tensorflow as tf
from evaluator.voceval import EvaluatorVOC

class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer):
    super().__init__(args, model, optimizer)

  def _get_loggers(self):
    super()._get_loggers()
    self.TESTevaluator = EvaluatorVOC(anchors=self.anchors,
                                      cateNames=self.labels,
                                      rootpath=self.dataset_root,
                                      score_thres=0.01,
                                      iou_thres=0.5,
                                      use_07_metric=False
                                      )
    self.logger_custom = ['AP@{}'.format(cls) for cls in self.labels] + ['mAP']

  # def _valid_epoch(self, multiscale, flip):
  #   results, imgs = super()._valid_epoch(multiscale, flip)
  #   return results, imgs
  #
  # def _train_epoch(self):
  #   for i, inputs in enumerate(self.train_dataloader):
  #     inputs = [tf.squeeze(input, axis=0) for input in inputs]
  #     img, _, _, _, _, *labels = inputs
  #     self.global_iter.assign_add(1)
  #     if self.global_iter.numpy() % 100 == 0:
  #       print(self.global_iter.numpy())
  #       for k, v in self.logger_losses.items():
  #         print(k, ":", v.result().numpy())
  #     self.train_step(img, labels)
  #
  #     if self.global_iter.numpy() % self.log_iter == 0:
  #       results, imgs = self._valid_epoch(multiscale=False,flip=False)
  #       with self.summarywriter.as_default():
  #         current_lr = self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
  #         tf.summary.scalar("learning_rate", current_lr, step=self.global_iter.numpy())
  #
  #         for k, v in zip(self.logger_voc, results):
  #           tf.summary.scalar(k, v, step=self.global_iter.numpy())
  #         for k, v in self.logger_losses.items():
  #           tf.summary.scalar(k, v.result(), step=self.global_iter.numpy())
  #         for i in range(len(imgs)):
  #           tf.summary.image("detections_{}".format(i), tf.expand_dims(tf.convert_to_tensor(imgs[i]), 0),
  #                            step=self.global_iter.numpy())
  #       self._reset_loggers()
  #   self.ckpt_manager.save(self.global_epoch)
