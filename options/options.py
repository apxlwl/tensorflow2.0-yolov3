import argparse
class Options():
  def __init__(self):
    parser = argparse.ArgumentParser(description='TRAIN yoloV3',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name',required=True)
    parser.add_argument('--dataset_name', default='COCO',required=True)
    parser.add_argument('--dataset_root',type=str)
    parser.add_argument('--pretrained_model',type=str,default='/home/gwl/PycharmProjects/mine/tf2-yolo3/checkpoints/darknet_coco')
    parser.add_argument('--gpu', default='0')
    # Optimization options
    parser.add_argument('--resume', default=None) #options:['load_darknet','load_yolov3',int]
    parser.add_argument('--total_epoch', type=int, default=200, help='Number of epochs to train.')

    parser.add_argument('--batch_size', type=int,default=12, help='Batch size for training.')
    parser.add_argument('--lr_initial', type=float, default=1e-4, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

    parser.add_argument('--gpu_ids', type=str,default=1, help='empty for CPU, other for GPU-IDs')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
    # Random seed
    # parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ignore_opti', action='store_true')

    # frequency_params
    parser.add_argument('--log_iter', type=int, default=5000)
    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--valid_batch', default=50,help='reduce the validation to save time')

    # important
    parser.add_argument('--freeze_darknet', action='store_true')
    parser.add_argument('--net_size', type=int ,default=416)
    #Test augmentation
    parser.add_argument('--fliptest',action='store_true')
    parser.add_argument('--multitest',action='store_true')

    self.opt = parser.parse_args()

