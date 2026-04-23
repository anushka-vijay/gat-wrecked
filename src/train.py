# Standardized training/eval loops
import torch
import torch.nn.functional as F


def train(model, features, adj, labels, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(features, adj)
    loss = F.nll_loss(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, features, adj, labels, mask):
    model.eval()
    out = model(features, adj)
    preds = out.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()
