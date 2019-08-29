import math
import time


def settings_summary(args):
    print('''
            General Settings:
                Dataset: {}
                Arch: {}  
            
            Hyper Parameter Settings: 
                N Class: {}
                Loss: {} 
                Optimizer: {}
                Epochs: {}
                Batch size: {}
                Learning rate: {}
                CUDA: {}
            '''.format(args.dataset,
                       args.arch,
                       args.n_class,
                       args.loss,
                       args.optimizer,
                       args.epochs,
                       args.batch_size,
                       args.lr,
                       args.gpu
                       ))

def error_ttc_vis(epoch,num_batches,batch_idx,now,epochs, loss_item,time_epcch):
    if epoch == 0:
        later = time.time()
        time_epcch = num_batches / (batch_idx + 1) * (later - now)

    remaining_epoch = epochs - epoch - (batch_idx + 1) / num_batches
    estimated_time = remaining_epoch * time_epcch
    hour = math.floor(estimated_time / 3600)
    min1 = estimated_time % 3600
    min2 = math.floor(min1 / 60)
    sec = (min1 % 60)
    print(
        'ep: [{0:.0f}/{1:.0f}] batch: [{2:.0f}/{3:.0f}] loss: {4:.6f} TTC: {5:.0f}h {6:.0f}m {7:.0f}s'.format(
            epoch + 1, epochs, batch_idx + 1,
            num_batches, loss_item, hour, min2, sec))




