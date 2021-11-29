# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/27 15:12
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：clustering.py
@IDE ：PyCharm

"""
import random

import torch
from torch import nn
import tools
from torch.multiprocessing import Pool


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
    def __init__(self, distance: str, seed_num: int, bandwidth: float, iteration: int, devices: list):
        """
        flat kernel is used in this implementation.
        :param distance:
        :param seed_num:
        :param bandwidth:
        :param iteration:
        """
        super(MeanShiftLayer, self).__init__()
        self.distance = tools.Distance(distance)
        self.seed_num = seed_num
        self.bandwidth = bandwidth
        self.iteration = iteration
        self.devices = devices

    def forward(self, x):
        """

        :param x: [samples, dimensions]
        :return:
        """
        with torch.no_grad():
            init_points = x[torch.randperm(len(x))[:self.seed_num]]
            init_points = [i[None] for i in init_points]
            if len(self.devices) == 1:
                return self.run(x, init_points, self.devices[0])
            else:
                pool = Pool(len(self.devices))
                _l = int(len(init_points)/len(self.devices))
                if _l*len(self.devices) != len(init_points):
                    left = len(init_points) - _l*len(self.devices)
                init_points_ = [init_points[i*_l:(i+1)*_l] for i in range(len(self.devices))]
                init_points_[-1].extend(init_points[-left:])
                data = [x for _ in range(len(self.devices))]
                centers,index_class, centers_similar, index_similar = pool.map(self.run, [data, init_points_, self.devices])
                return centers,index_class, centers_similar, index_similar

    def run(self, x, init_points, device):
        x = x.to(device)
        init_points = [i.to(device) for i in init_points]
        center_points = init_points
        index_class = list(range(len(init_points)))
        for it in range(self.iteration):
            for index_c, point in enumerate(center_points):
                dis = self.distance(x, point, dim=1)
                index = torch.where(dis < self.bandwidth)
                center_points[index_c] = x[index].mean(dim=0, keepdim=True)
                index_class[index_c] = index[0]  # torch.where returns a tuple
        # get index and features that most similar to centers.
        index_similar, center_similar = [], []
        for index, center in zip(index_class, center_points):
            argmin = self.distance(x[index], center, dim=1).argmin()
            index_similar.append(index[argmin])
            center_similar.append(x[index[argmin]])
        center_points = torch.cat(center_points, dim=0)
        center_similar = torch.stack(center_similar, dim=0)
        index_similar = torch.stack(index_similar)
        return center_points, index_class, center_similar, index_similar


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    from sklearn.datasets import make_blobs

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    centers = [[1, 1], [-1, -1], [1, -1]]
    N, D, K = 64000, 2, 100
    bandwidth = 1
    iteration = 10
    x, _ = make_blobs(n_samples=N, centers=K, cluster_std=0.6, random_state=seed)
    x = torch.from_numpy(x)
    torch.manual_seed(0)
    # x = 0.7 * torch.randn(N, D) + 0.3
    # x = 0.7 * torch.randn(N, D) + 0.3
    # kmeans_euc = KMeansLayer(K, 10, distance='euclidean', save_memory=False)  # set to cuda if necessary,
    devices = [torch.device('cuda:0')]
    meanshift = MeanShiftLayer('euclidean', K*2, bandwidth, iteration, devices)
    start = time.time()
    # avg_c, cl, c, index = kmeans_euc(x)
    avg_c, cl, c, index = meanshift(x)
    print('Time:', time.time()-start, 's')
    cl = torch.cat(cl, dim=0)
    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=30000 / len(x), cmap="tab10")
    plt.scatter(c[:, 0].cpu(), c[:, 1].cpu(), c="black", s=50, alpha=0.8)
    plt.scatter(avg_c[:, 0].cpu(), avg_c[:, 1].cpu(), c="red", s=50, alpha=0.8)
    for i in c:
        circle = plt.Circle((i[0], i[1]), bandwidth, color='y', fill=False)
        plt.gcf().gca().add_artist(circle)
    plt.tight_layout()
    plt.show()