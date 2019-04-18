##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################

from .coco import get_dataset as get_COCO
from .pascal import get_dataset as get_VOC
from .image import makeImgPyramids
from .bbox import bbox_flip