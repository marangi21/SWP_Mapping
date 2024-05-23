import torch

def channel_means_stds(dataset):
    dataset = torch.stack(list(img for img in dataset.images))
    means = []
    stds = []
    for i in range(11):
        channel = dataset[:, i, :, :]
        mean = channel.mean()
        std = channel.std()
        means.append(mean)
        stds.append(std)
    channel_means = [t.tolist() for t in means]
    channel_stds = [t.tolist() for t in stds]
    return channel_means, channel_stds