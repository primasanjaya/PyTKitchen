import call_dataset


def train_execution(args, model):


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    data_loader = call_dataset(args)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target, requires_grad=False)
        optimizer.zero_grad()
        if args.with_reconstruction:
            output, probs = model(data, target)
            reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
            margin_loss = loss_fn(probs, target)
            loss = reconstruction_alpha * reconstruction_loss + margin_loss
        else:
            output, probs = model(data)
            loss = loss_fn(probs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))