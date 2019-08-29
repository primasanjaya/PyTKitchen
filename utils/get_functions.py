from model.capsulenet.capsnet import *
import torch.nn as nn
from dataset import *
import torch.utils.data as data
from torch import optim


def get_model(args):
    if args.arch == 'capsnet':
        model = CapsNet(args.routing_iterations)

    elif args.arch == 'capsnetrecon':
        model = ReconstructionNet(args.routing_iterations,args.n_class)

    return model


def get_loss(args):
    if args.loss == 'bce':
        criterion = nn.BCELoss()
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'customcapsnet':
        criterion = MarginLoss(0.9, 0.1, 0.5)
    elif args.loss == 'customcapsnetrecon':
        criterion = MarginLoss(0.9, 0.1, 0.5)
    else :
        criterion = None
    return criterion


def get_dataloader(args):
    if args.dataset == 'splice':
        dataloader_class = Splice(args.train_dir)
    else:
        dataloader_class = None

    data_loader = data.DataLoader(dataset=dataloader_class, num_workers=1, batch_size=args.batch_size, shuffle=True)

    return data_loader

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0.0005)
    else:
        optimizer = None

    return optimizer

