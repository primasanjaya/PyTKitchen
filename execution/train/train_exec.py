from utils.get_functions import *
from utils.visualization import *
from utils.saveload_hdd import *

import math
import time


def train_execution(args, model):
    #setup
    optimizer = get_optimizer(args,model)
    loss_fn = get_loss(args)
    data_loader, n_instance = get_dataloader(args)
    model.train()

    #additional variables
    settings_summary(args)
    num_batches = math.ceil(n_instance / args.batch_size)

    for epoch in range(args.epochs):
        time_epoch = 0
        if epoch > 0:
            later = time.time()
            time_epoch = later - now
        now = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            if args.gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.loss=='customcapsnetrecon':
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = 0.0005 * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            #pdb.set_trace()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                error_ttc_vis(epoch,num_batches,batch_idx,now,args.epochs, loss.item(),time_epoch)

        save_weight(args,model.module.state_dict(),epoch)
