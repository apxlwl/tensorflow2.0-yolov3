from trainers.base_trainer import BaseTrainer
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
    self.logger_custom = ['mAP']+['AP@{}'.format(cls) for cls in self.labels]