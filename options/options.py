import os, sys, time, pdb, random, argparse
class Options():
    def __init__(self, model_names):
        parser = argparse.ArgumentParser(description='Train Style Aggregated Network',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--config_path',default='./configs/face.json')
        # Optimization options
        parser.add_argument('--resume',default=None)
        parser.add_argument('--total_epoch', type=int, default=300, help='Number of epochs to train.')
        parser.add_argument('--batch_size', type=int, help='Batch size for training.')
        parser.add_argument('--learning_rate', type=float, help='The Learning Rate.')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')


        parser.add_argument('--gpu_ids', type=str, help='empty for CPU, other for GPU-IDs')
        parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
        # Random seed
        # parser.add_argument('--manualSeed', type=int, help='manual seed')
        parser.add_argument('--debug',action='store_true')
        parser.add_argument('--evaluate',action='store_true')
        parser.add_argument('--ignore_opti',action='store_true')

        #frequency_params
        parser.add_argument('--log_iter',type=int,default=600)
        parser.add_argument('--save_iter',type=int,default=5000)
        parser.add_argument('--save_best',action='store_true')
        parser.add_argument('--load_best',action='store_true')
        parser.add_argument('--do_valid',default=True)
        parser.add_argument('--adjust_list',type=str)
        self.opt = parser.parse_args()

