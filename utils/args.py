import argparse
import torch
from utils.string_processing import *

def fix_dir(args):
    args.l_w_dir = ensure_path(args.l_w_dir)
    args.train_dir = ensure_path(args.train_dir)
    args.train_gt = ensure_path(args.train_gt)
    args.s_w_dir = ensure_path(args.s_w_dir)
    args.pred_dir = ensure_path(args.pred_dir)
    args.result_dir = ensure_path(args.result_dir)
    args.val_dir = ensure_path(args.val_dir)
    args.val_gt = ensure_path(args.val_gt)
    args.s_w_name = ensure_terminator(args.s_w_name, ".pth")

    return args

def load_hyperparams(args):
    if not args.no_gpu and torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    args.gpu = use_gpu
    args.no_gpu = not use_gpu

    if args.loss == None:
        if args.arch =='capsnetrecon':
            args.loss = 'customcapsnetrecon'

    return args

def get_args():
    parser = argparse.ArgumentParser(description='My PyTorch Kitchen')

    # DATASET
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset')
    # MODEL
    parser.add_argument('--arch', type=str, default='general',
                        help='architecture')
    # DIRECTORY
    parser.add_argument('--train-dir', type=str, default=None,
                        help='training directory')
    parser.add_argument('--train-gt', type=str, default=None,
                        help='training groundtruth')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='validation directory')
    parser.add_argument('--val-gt', type=str, default=None,
                        help='validation groundtruth')
    parser.add_argument('--pred-dir', type=str, default=None,
                        help='training directory')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='predict groundtruth')

    parser.add_argument('--s-w-dir', type=str, default=None,
                        help='save weight directory')
    parser.add_argument('--s-w-name', type=str, default=None,
                        help='save filename')
    parser.add_argument('--l-w-dir', type=str, default=None,
                        help='load weight directory')
    parser.add_argument('--l-w-name', type=str, default=None,
                        help='load filename')

    # HYPER PARAMETERS
    parser.add_argument('--n-class', type=int, default=1,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default=None,
                        help='')
    parser.add_argument('--optimizer', type=str, default=None,
                        help='')

    # OTHERS
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--pred', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)

    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='enable CUDA training')

    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--epoch-done', type=int, default=0,
                        help='how many training epochs have done')

    args = parser.parse_args()

    return args
