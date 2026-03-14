import torch
from torch import nn

import matplotlib.pyplot as plt


def solution_1_forward(ae, x):
    out = ae.encoder(x)
    out = ae.decoder(out)
    out = torch.reshape(out, (-1, 1, 28, 28))
    return out


def solution_2_train_one_step(model, batch, optimizer):
    optimizer.zero_grad()
    reconstruction = model(batch)
    loss = nn.functional.mse_loss(batch, reconstruction)
    loss.backward()
    optimizer.step()
    return loss


def solution_3_reconstruct_images(ae, test_loader, num_images=5):
    test_batch = next(iter(test_loader))[0][:num_images]
    reconstructions = ae(test_batch).squeeze().detach()
    fig, ax = plt.subplots(num_images, 2)
    fig.set_size_inches(5, 10 * (num_images // 5))
    for i in range(num_images):
        ax[i][0].imshow(test_batch[i][0])
        ax[i][1].imshow(reconstructions[i])
        for a in ax[i]:
            a.axis('off')


def solution_4_anomaly_score(x, reconstruction):
    return nn.functional.mse_loss(x, reconstruction,
                                  reduction="none").mean(dim=(1, 2, 3))


def solution_5_latent_scatter_plot(ae, loader):
    images, labels = [], []

    for i, (x, y) in enumerate(loader):
        images.append(x)
        labels.append(y)
        if i > 3:
            break

    images = torch.cat(images)
    labels = torch.cat(labels)

    latents = ae.encoder(images)

    plt.scatter(*latents.T.detach(), c=labels, cmap="Set10")
    plt.xlabel("$l_1$")
    plt.ylabel("$l_2$")
    plt.colorbar()
    plt.show()
