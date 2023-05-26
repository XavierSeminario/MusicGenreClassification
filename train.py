import tqdm
import numpy as np

def train(model, device, train_loader, optimizer, criterion):
    losses = []
    model.train()
    t = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    t.set_description('Train')
    for batch_idx, (data, target) in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #backpropagation
        #data = data.resize_(data.size()[0], 64,64).float()
        output = model(data)
        #print(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        t.set_postfix(loss=loss.item())

    return np.mean(losses)