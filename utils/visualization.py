
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
                       args.optimizer,
                       args.epochs,
                       args.batch_size,
                       args.lr,
                       args.no_cuda
                       ))


