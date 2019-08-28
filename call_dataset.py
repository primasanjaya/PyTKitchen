from dataset.splice import *
from utils.args import *

from torchvision import datasets, transforms
from utils.args import get_args
from dataset.splice import *


def call_dataloader(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Pad(2), transforms.RandomCrop(28),
                               transforms.ToTensor()
                           ])), batch_size=args.batch_size, shuffle=True, **kwargs)
    if args.dataset == 'splice':
        data_loader = Splice(args)


if __name__ == '__main__':
    args = get_args()

    args.train_dir = 'E:/data/zother/splice/train/'
    args.val_dir = 'E:/data/zother/splice/val/'
    args.dataset = 'mnist'

    data_loader = call_dataloader(args)
    pdb.set_trace()

    img, target = data_loader.__getitem__(0)
