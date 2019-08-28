
from model.capsulenet.capsnet import *

def get_model(args):
    if args.arch == 'capsnet':
        model = CapsNet(args.routing_iterations)