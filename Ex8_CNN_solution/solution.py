"Solution for HW 8 (CNNs)"

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.utils import make_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def solution_1_train_and_val_split(dataset, train_split, val_split):
    if train_split + val_split == 1:
        train_set, val_set = random_split(dataset, [train_split, val_split])
    else:
        train_set, val_set, _ = random_split(
            dataset, [train_split, val_split, (1 - train_split - val_split)])

    return train_set, val_set


def solution_2_create_cnn(input_shape: tuple, cnn_channels: list[int],
                          n_classes: int, create_mlp: Callable):
    input_channels, input_height, input_width = input_shape

    hidden_conv_layers = []
    for i in range(len(cnn_channels) - 1):
        hidden_conv_layers.extend([
            nn.Conv2d(cnn_channels[i],
                      cnn_channels[i + 1],
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        ])

    conv_output_shape = (cnn_channels[-1],
                         input_height // (2**(len(cnn_channels))),
                         input_width // (2**(len(cnn_channels))))

    projection_head = create_mlp(conv_output_shape, [], n_classes)

    model = nn.Sequential(
        # Convolutional part
        nn.Conv2d(input_channels,
                  cnn_channels[0],
                  kernel_size=5,
                  stride=1,
                  padding=2),
        nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        *hidden_conv_layers,
        # Projection head
        projection_head)

    return model


def solution_3_train_n_models(n_models: int, create_mlp: Callable,
                              create_cnn: Callable, input_shape: tuple,
                              mlp_dims: list[int], cnn_channels: list[int],
                              n_classes: int, lr: float,
                              train_loader: DataLoader, val_loader: DataLoader,
                              n_epochs: int, train_fn: Callable):
    mlps = []
    cnns = []

    for i in range(n_models):
        mlp = create_mlp(input_shape, mlp_dims, n_classes)
        optimizer_mlp = Adam(mlp.parameters(), lr=lr)

        cnn = create_cnn(input_shape, cnn_channels, n_classes)
        optimizer_cnn = Adam(cnn.parameters(), lr=lr)

        print(f"Training MLP no. {i} ...")
        losses_mlp = train_fn(mlp,
                              train_loader,
                              val_loader,
                              optimizer=optimizer_mlp,
                              loss_fn=F.cross_entropy,
                              n_epochs=n_epochs,
                              validate_every=1)

        print(f"Training CNN no. {i} ...")
        losses_cnn = train_fn(cnn,
                              train_loader,
                              val_loader,
                              optimizer=optimizer_cnn,
                              loss_fn=F.cross_entropy,
                              n_epochs=n_epochs,
                              validate_every=1)

        plt.plot(losses_mlp[0], ".--b", label="Train loss (MLP)")
        plt.plot(losses_mlp[2], losses_mlp[1], ".-b", label="Val loss (MLP)")

        plt.plot(losses_cnn[0], ".--r", label="Train loss (CNN)")
        plt.plot(losses_cnn[2], losses_cnn[1], ".-r", label="Val loss (CNN)")

        plt.legend()
        plt.show()

        mlps.append(mlp)
        cnns.append(cnn)

    return mlps, cnns


def solution_4a_accuracy_boxplots(get_accuracy_fn: Callable,
                                  mlps: list[nn.Module], cnns: list[nn.Module],
                                  dataloader: DataLoader):
    mlp_accs = [get_accuracy_fn(mlp, dataloader) for mlp in mlps]
    cnn_accs = [get_accuracy_fn(cnn, dataloader) for cnn in cnns]

    plt.boxplot([mlp_accs, cnn_accs])
    plt.xticks([1, 2], ["mlp", "cnn"])
    plt.ylabel("Accuracy")
    plt.show()

    print(f"MLP mean: {np.mean(mlp_accs)}, CNN mean: {np.mean(cnn_accs)}")


def solution_4b_count_params(model: nn.Module):
    total = 0
    for n, p in model.named_parameters():
        n_params = p.data.numel()
        total += n_params
    return total


def solution_4c_plot_feature_maps(model: nn.Module, image: torch.Tensor):
    x = image.to(DEVICE)
    print("Input image")
    plt.imshow(x.detach().cpu().squeeze().permute(1, 2, 0))
    plt.show()
    for i, layer in enumerate(model):
        x = layer(x)
        if i in [2, 5, 8]:
            print(f"Layer {i} ({layer})")
            activation_grid = make_grid(
                x.detach().cpu().squeeze()[:, None, :] /
                x.detach().cpu().max(),
                pad_value=1,
                padding=1)
            plt.imshow(activation_grid.permute(1, 2, 0).squeeze())
            plt.axis("off")
            plt.show()
