import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from IPython import get_ipython


def get_global_var_from_notebook(name: str):
    ipython = get_ipython()

    if not ipython:
        raise EnvironmentError(
            "This function should be used inside a jupyter notebook")

    variable = ipython.user_ns.get(name, None)
    if variable is not None:
        return variable
    return None


def solution_1a_init(dims: list[int]):
    modulelist = nn.ModuleList([nn.Flatten()])
    for in_features, out_features in zip(dims[:-2], dims[1:]):
        modulelist.extend([nn.Linear(in_features, out_features), nn.ReLU()])
    modulelist.append(nn.Linear(dims[-2], dims[-1]))
    return modulelist


def solution_1a_forward(inp: torch.Tensor, layers: nn.ModuleList):
    out = inp
    for l in layers:
        out = l(out)
    return out


def solution_1b_simplefixedmlp(model_cls):
    return model_cls()


def solution_1b_simplemlp(model_cls):
    return model_cls(dims=[28 * 28, 32, 32, 32, 10])


def solution_1b_sequential():
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 32), nn.ReLU(),
                         nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, 10))


def solution_2a_train(model, batch, labels, optimizer, loss_fn):
    optimizer.zero_grad()
    preds = model(batch)
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def solution_2b_optimizers(models):
    return [Adam(model.parameters(), lr=0.04) for model in models]


def solution_2b_train(train_fn, models, optimizers, loss_fn, n_epochs=20):
    assert len(models) == len(optimizers)

    train_loader: DataLoader = get_global_var_from_notebook("train_loader")

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        print(f"Training model {i+1} ...")

        losses = train_fn(model, train_loader, optimizer, loss_fn, n_epochs)

        filter_size = 50
        losses_np = np.array(losses)
        smoothing_filter = np.ones(filter_size) / filter_size

        losses_smooth = np.convolve(losses_np, smoothing_filter, mode="valid")

        plt.plot(losses_smooth, label=f"Model {i+1}")
    plt.legend(loc="upper right")
    plt.show()


def solution_3a_compute_accuracy(model, batch, labels):
    logits = model(batch)
    preds = torch.argmax(logits, dim=-1)
    return torch.sum(preds == labels).item()


def solution_3b_accuracy_per_class(model, dataloader):
    n_correctly_classified_per_class = {k: 0 for k in range(10)}
    n_samples_per_class = {k: 0 for k in range(10)}
    for batch, labels in dataloader:
        logits = model(batch)
        preds = torch.argmax(logits, dim=-1)
        correctly_classified = labels == preds

        for cls, correct in zip(labels.flatten().tolist(),
                                correctly_classified.flatten().tolist()):
            n_samples_per_class[cls] += 1
            if correct:
                n_correctly_classified_per_class[cls] += 1

    accuracy_per_class = {
        k: n_correctly_classified_per_class[k] / n_samples_per_class[k]
        for k in range(10)
    }
    return accuracy_per_class
