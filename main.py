from __future__ import print_function

from torch.optim import lr_scheduler
from torch.autograd import Variable

import pdb

from utils.args import get_args
from dataset.dataset import *

from utils import *
from execution.train.train_exec import *


args = get_args()

model = get_model(args)



if args.train:
    print('Training')
    train_execution(args,model)

elif args.val:
    print('todo')
elif args.pred:
    print('todo')
