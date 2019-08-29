from utils.get_functions import *
from utils.visualization import settings_summary
import math
import time

def train_execution(args, model):


    optimizer = get_optimizer(args,model)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = get_loss(args)
    data_loader = get_dataloader()
    model.train()

    settings_summary(args)

    num_batches = math.ceil(len(data_loader) / args.batch_size)

    for epoch in range(args.epochs):
        if epoch > 0:
            later = time.time()
            time_epoch = later - now
        now = time.time()

        for batch_idx, (data, target) in enumerate(data_loader):
            if args.cuda:
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
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                if epoch == 0:
                    later = time.time()
                    time_epoch = num_batches / (batch_idx + 1) * (later - now)
                remaining_epoch = args.epochs - epoch - (batch_idx + 1) / num_batches
                estimated_time = remaining_epoch * time_epoch


                hour = math.floor(estimated_time / 3600)
                min1 = estimated_time % 3600
                min2 = math.floor(min1 / 60)
                sec = (min1 % 60)
                print(
                    'ep: [{0:.0f}/{1:.0f}] batch: [{2:.0f}/{3:.0f}] loss: {4:.6f} TTC: {5:.0f}h {6:.0f}m {7:.0f}s'.format(
                        epoch + 1, args.epochs, batch_idx + 1,
                        num_batches, loss.item(), hour, min2, sec))