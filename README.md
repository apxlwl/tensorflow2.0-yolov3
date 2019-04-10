#tf2-yolo3 
##Introduction
A Tensorflow2.0 implementation of YOLOv3

##Quick Start 
1. Download yolov3.weights and darknet53.conv.74 from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Download COCO dataset
3. Modify the dataset root and weights root in the config file ("./configs/coco.json")
4. Do evaluation (the result is in "./results/darknet_416.txt"):
```
python main.py --resume load_yolov3 --do_test
```

##Training
1. run the following command to start training
```
python main.py --experiment_name dummy_name
```


2. The ckpt file will be stored in the ./checkpoints/dummy_name named with epoch number and use --resume xyz to resume from the xyz epoch

3. The summary file is stored in the  ./summary/dummy_name and origanized like [TF-ObjectDection-API](https://github.com/tensorflow/models/tree/master/research/object_detection)



##TODO
-[ ] Update performance
-[ ] Support distribute training
-[ ] Support Custom dataset  
##Reference
[gluon-cv](https://github.com/dmlc/gluon-cv)

[tf-eager-yolo3](https://github.com/penny4860/tf-eager-yolo3)

[keras-yolo3](https://github.com/qqwweee/keras-yolo3)