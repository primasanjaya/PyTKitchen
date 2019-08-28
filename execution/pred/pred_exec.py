def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data, volatile=True), Variable(target)

        if args.with_reconstruction:
            output, probs = model(data, target)
            reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).data[0]
            test_loss += loss_fn(probs, target, size_average=False).item()
            test_loss += reconstruction_alpha * reconstruction_loss
        else:
            with torch.no_grad():
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).item()

        pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss