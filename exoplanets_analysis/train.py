import torch

def train_step(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    for batch_idx, (target, data) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # forward
        output = model(data)
        loss = criterion(output, target)

        # backward and update
        loss.backward()
        optimizer.step()

    print('\n######\nTrain Epoch: {}'.format(epoch))
    # show loss of last batch of this epoch
    output = model(data)
    loss = criterion(output, target)
    print('Mean loss: {:.6f}'.format(loss / len(data)) )



