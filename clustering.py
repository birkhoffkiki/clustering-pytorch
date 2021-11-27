# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/27 15:12
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：clustering.py
@IDE ：PyCharm

"""

import torch
from torch import nn
import tools


class KMeansLayer(nn.Module):
    def __init__(self, centers, iteration, distance='euclidean', save_memory=True):
        super(KMeansLayer, self).__init__()
        self.distance = distance
        self.similarity = tools.Similarity(distance, save_memory)
        self.clusters = centers
        self.iteration = iteration

    def forward(self, data):
        avg_center, classes, centers, index = self.kmeans(data, self.clusters, self.iteration)
        return avg_center, classes, centers, index

    def kmeans(self, data: torch.Tensor, clusters: int, iteration: int):
        """
        :param data: [samples, dimension]
        :param clusters: the number of centers
        :param iteration: total iteration time
        :return: [average_center, class_map, center, index]
        """
        with torch.no_grad():
            N, D = data.shape
            c = data[torch.randperm(N)[:clusters]]
            for i in range(iteration):
                a = self.similarity(data, c, mode=1)
                c = torch.stack([data[a == k].mean(0) for k in range(clusters)])
                nanix = torch.any(torch.isnan(c), dim=1)
                ndead = nanix.sum().item()
                c[nanix] = data[torch.randperm(N)[:ndead]]
            # get centers (not average centers) and index
            index = self.similarity(data, c, mode=0)
            center = data[index]
            avg_center = c
        return avg_center, a,  center, index


class MeanShiftLayer(nn.Module):
    def __init__(self):
        super(MeanShiftLayer, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    torch.manual_seed(0)
    N, D, K = 64000, 2, 60
    x = 0.7 * torch.randn(N, D) + 0.3
    x = 0.7 * torch.randn(N, D) + 0.3
    kmeans_euc = KMeansLayer(K, 10, distance='euclidean', save_memory=False)  # set to cuda if necessary,
    start = time.time()
    avg_c, cl, c, index = kmeans_euc(x)
    print('Time:', time.time()-start, 's')
    plt.subplot(121)
    plt.suptitle('Euclidean distance')
    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=cl.cpu(), s=30000 / len(x), cmap="tab10")
    plt.scatter(c[:, 0].cpu(), c[:, 1].cpu(), c="black", s=50, alpha=0.8)
    plt.scatter(avg_c[:, 0].cpu(), avg_c[:, 1].cpu(), c="red", s=50, alpha=0.8)
    plt.axis([-2, 2, -2, 2])
    plt.tight_layout()
    plt.show()