# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/27 16:21
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：speed_test.py
@IDE ：PyCharm

"""
from clustering import KMeansLayer
import torch
import time


def k_means(device, save_memory):
    torch.manual_seed(0)
    test_config = [(10000, 2, 100), (10000, 2, 1000), (100000, 2, 100), (100000, 2, 1000), (20000, 200, 400)]
    for cfg in test_config:
        N, D, K = cfg
        x = 0.7 * torch.randn(N, D) + 0.3
        kmeans_euc = KMeansLayer(K, 10, distance='euclidean', save_memory=save_memory)  # set to cuda if necessary,
        if device == 'GPU':
            x = x.cuda()
            kmeans_euc = kmeans_euc.cuda()
        start = time.time()
        avg_c, cl, c, index = kmeans_euc(x)
        end = time.time()
        print('Save memory:{}, {}, {}, time is {} s:'.format(save_memory, device,
                                                         cfg, end-start))


if __name__ == '__main__':
    k_means('CPU', True)
    k_means('GPU', True)
    k_means('CPU', False)
    k_means('GPU', False)
