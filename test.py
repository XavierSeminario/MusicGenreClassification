import tqdm
import numpy as np
import wandb
import torch

#comprovar si funciona ok, not sure
def test(model, device, test_loader, criterion):
    losses = []
    model.eval()
    all_preds = []
    
    t = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    t.set_description('Test')
    with torch.no_grad():
        correct, total = 0, 0
        for batch_idx, (data, target) in t: #iterem sobre les dades
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            losses.append(loss.item())
            t.set_postfix(loss=loss.item())
            all_preds.extend(predicted.detach().cpu().numpy())
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})
    return np.mean(losses), all_preds
