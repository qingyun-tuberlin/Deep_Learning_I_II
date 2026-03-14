import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def solution_1a_create_model(hidden_dims: list[int]) -> nn.Module:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_dims[0]),
        nn.ReLU(),
        *[
            nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU())
            for in_feats, out_feats in zip(hidden_dims[:-1], hidden_dims[1:])
        ],
        nn.Linear(hidden_dims[-1], 10),
    )
    return model


def solution_2b_evaluate(
    model: nn.Module, dataloader: DataLoader
) -> tuple[float, float]:
    losses = []
    accuracies = []

    model.eval()  # TODO: explain in markdown

    with torch.no_grad():  # TODO: explain in markdown
        for batch, labels in dataloader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(batch)

            loss = cross_entropy(preds, labels)
            losses.append(loss.item())

            accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels)
            accuracies.append(accuracy.item() / len(labels))

    model.train()  # TODO: explain in markdown

    avg_loss = sum(losses) / len(losses)
    avg_acc = sum(accuracies) / len(accuracies)

    return avg_loss, avg_acc


def solution_2a_train_one_epoch(model, dataloader, optimizer):
    losses = []
    for batch, labels in dataloader:
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        preds = model(batch)
        loss = cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def solution_2c_plot_train_and_val_loss(
    epoch: int,
    train_losses: list[float],
    val_losses: list[float],
    val_acc: float,
) -> None:
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.axhline(y=min(val_losses), c="r", linestyle="--")
    plt.legend(loc="upper right")
    plt.title(
        f"Epoch {epoch} | train loss = {train_losses[-1]:.2f} | val loss = {val_losses[-1]:.2f} | val accuracy = {val_acc:.2f}"
    )
    plt.show()


def solution_3a_save_model_if_improved(
    model, epoch, train_losses, val_losses, val_acc, model_name
):
    if val_losses[-1] == min(val_losses):
        print(
            f"Found new best model at epoch {epoch} with a validation loss of {val_losses[-1]:.4f}"
        )
        torch.save(model.state_dict(), model_name)


def solution_4a_create_model_with_dropout(hidden_dims: list[int], p: float):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_dims[0]),
        nn.ReLU(),
        nn.Dropout(p),
        *[
            nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU(), nn.Dropout(p))
            for in_feats, out_feats in zip(hidden_dims[:-1], hidden_dims[1:])
        ],
        nn.Linear(hidden_dims[-1], 10),
    )
    return model
    