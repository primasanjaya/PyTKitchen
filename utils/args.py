import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='My PyTorch Kitchen')

    # DATASET
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset')
    # MODEL
    parser.add_argument('--arch', type=str, default='capsnet',
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
    parser.add_argument('--pred-gt', type=str, default=None,
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

    # OTHERS
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--pred', action='store_true', default=False)


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = parser.parse_args()

    return args
