import tqdm
import numpy as np

#comprovar si funciona ok, not sure
def test(model, device, test_loader, criterion):
    losses = []
    model.eval()
    t = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    t.set_description('Test')
    with torch.no_grad():
        for batch_idx, (data, target) in t: #iterem sobre les dades
            data, target = data.to(device), target.to(device)
            #data = data.resize_(data.size()[0], 64, 64).float() #redimensió de les dades
            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            t.set_postfix(loss=loss.item())

    return np.mean(losses)
