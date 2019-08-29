
import pdb

from utils.args import get_args
import os
from dataset.dataset import *

from utils import *
from execution.train.train_exec import *


args = get_args()
args = fix_dir(args)
model = get_model(args)

if args.train:
    print('Training')
    os.makedirs(os.path.dirname(args.s_w_dir), exist_ok=True)
    train_execution(args,model)
elif args.val:
    print('todo')
elif args.pred:
    print('todo')
