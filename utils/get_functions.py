from model.capsulenet.capsnet import *
from model.birnn.birnn import *

import torch.nn as nn
import torch.utils.data as data
from torch import optim
from utils.saveload_hdd import *

from dataset.mnistbirnn import MNISTBiRNN
from dataset.splice import Splice
from torchvision import transforms,datasets
from model.segcaps.segcaps import SegCaps
from utils.custom_loss import *


def get_model(args):
    if args.arch == 'capsnet':
        model = CapsNet(args.routing_iterations)
    elif args.arch == 'capsnetrecon':
        model = CapsNet(args.routing_iterations)
        rec = ReconstructionNet(16,args.n_class)
        model = CapsNetWithReconstruction(model, rec)
    elif args.arch == 'birnn':
        model = BiRNN(input_size=28, hidden_size=128, num_layers=2, num_classes=args.n_class)
    elif args.arch == 'segcaps':
        model = SegCaps()

    model = nn.DataParallel(model)

    if args.load:
        state = load_weight(args)
        weight = state['weight']

        model_state = model.module.state_dict()
        model_state.update(weight)
        model.module.load_state_dict(model_state)

    return model


def get_loss(args):
    if args.loss == 'bce':
        criterion = nn.BCELoss()
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'customcapsnet':
        criterion = CapsNetMarginLoss(0.9, 0.1, 0.5)
    elif args.loss == 'customcapsnetrecon':
        criterion = CapsNetReconLoss(0.9, 0.1, 0.5)
    else :
        criterion = None
    return criterion


def get_dataloader(args):
    if args.dataset == 'splice':
        dataloader_class = Splice(args.train_dir)
    elif args.dataset == 'mnist':
        data_transform = transforms.Compose([transforms.Pad(2), transforms.RandomCrop(28),transforms.ToTensor()])
        dataloader_class = datasets.MNIST('../data', train=True, download=True, transform=data_transform)
    elif args.dataset == 'mnistbirnn':
        data_transform = transforms.Compose([transforms.Pad(2), transforms.RandomCrop(28),transforms.ToTensor()])
        dataloader_class = MNISTBiRNN('../data', train=True, transform=data_transform)
    else:
        dataloader_class = None

    data_loader = data.DataLoader(dataset=dataloader_class, num_workers=1, batch_size=args.batch_size, shuffle=True)

    return data_loader, len(dataloader_class)

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005)
    else:
        optimizer = None

    return optimizer

