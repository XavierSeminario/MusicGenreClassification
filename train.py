import tqdm
import numpy as np
import wandb

def train(model, device, train_loader, optimizer, criterion, epoch):
    
    wandb.watch(model, criterion, log="all", log_freq=10)

    losses = []
    model.train()
    example_ct = 0  

    t = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    t.set_description('Train')
    for batch_idx, (data, target) in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #backpropagation
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        example_ct +=  len(data)

        losses.append(loss.item())
        t.set_postfix(loss=loss.item())

        if ((batch_idx + 1) % 25) == 0:
               train_log(loss, example_ct, epoch)

    return np.mean(losses)


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")