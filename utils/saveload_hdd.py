import torch

def save_weight(args,weight,epoch):
    state = {'n_class': args.n_class, 'arch': args.arch, 'epoch_done': args.epoch_done + epoch + 1,
             'task': args.task, 'weight': weight, 'loss': args.loss}
    torch.save(state, args.s_w_dir + args.s_w_name + '-{0:.0f}.pth'.format(args.epoch_done + epoch))
    print('Checkpoint {} saved !'.format(epoch + 1))

def load_weight(args):
    state = list()
    state = torch.load(args.l_w_dir + args.l_w_name)
    return state
