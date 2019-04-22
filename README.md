# tf2-yolo3 

## Introduction
A Tensorflow2.0 implementation of YOLOv3

## Quick Start 
1. Download yolov3.weights and darknet53.conv.74 from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Download COCO dataset
3. Modify the dataset root and weights root in the config file
```
python main_coco.py --resume load_yolov3 --do_test --net_size 608 --batch_size 8
```

## Training
1. run the following command to start training
```
python main_voc.py/main_coco.py --resume load_darknet --net_size 480 --batch_size 8
```


2. The ckpt file will be stored in the ./checkpoints/dummy_name named with epoch number and use --resume xyz to resume from the xyz epoch


## Visualization
The Tensorboard is origanized like [TF-ObjectDection-API](https://github.com/tensorflow/models/tree/master/research/object_detection)
![GT VS Prediction across time]()
## Performance 
| Model | Initial weight | basic resolution |VOC2007 Test(mAP) |
| ------ | ------ | ------ | ------ |
| YoloV3 | Darknet53 | 512|0.7796 |

## Supported Attributes
- [x] Data agumentation  
- [x] Multi-scale Training 
- [x] Multi-scale Testing(including flip)
- [x] Focal loss  
- [ ] ....
## TODO
- [x] Update VOC performance
- [ ] Update COCO performance
- [ ] Support distribute training
- [ ] Support Custom dataset  

## Reference
[gluon-cv](https://github.com/dmlc/gluon-cv)

[tf-eager-yolo3](https://github.com/penny4860/tf-eager-yolo3)

[keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[stronger-yolo](https://github.com/Stinky-Tofu/Stronger-yolo)