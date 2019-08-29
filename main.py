
if __name__ == '__main__':

    from utils.args import get_args
    import os

    from utils import *
    from execution.train.train_exec import *


    args = get_args()
    args = fix_dir(args)
    args = load_hyperparams(args)

    model = get_model(args)

    if args.train:
        print('Training')
        os.makedirs(os.path.dirname(args.s_w_dir), exist_ok=True)
        train_execution(args,model)
    elif args.val:
        print('todo')
    elif args.pred:
        print('todo')
