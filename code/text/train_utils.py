
from tqdm.notebook import tqdm
from tqdm import tqdm
import torch
from sklearn import metrics
import gc
import torch.nn as nn

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(args, model, optimizer, training_loader, epoch):
    model.train()
    i = 0
    for data in tqdm(training_loader):
        ids = data['ids'].to(args.device, dtype = torch.long)
        mask = data['mask'].to(args.device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(args.device, dtype = torch.long)
        targets = data['targets'].to(args.device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i%1000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        del token_type_ids, ids, mask, outputs, loss, data
        gc.collect()
        i += 1
        
    return model

def validation(args, model, epoch, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids = data['ids'].to(args.device, dtype = torch.long)
            mask = data['mask'].to(args.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(args.device, dtype = torch.long)
            targets = data['targets'].to(args.device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            del targets, ids, mask, outputs, data
            gc.collect()

    return fin_outputs, fin_targets
