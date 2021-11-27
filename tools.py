# -*- coding: utf-8 -*-
"""
@Time ： 2021/11/27 15:12
@Auth ： majiabo, majiabo@hust.edu.cn
@File ：tools.py
@IDE ：PyCharm

"""
import torch


class Similarity:
    supported_distance = ['euclidean', 'cosine']

    def __init__(self, distance: str, save_memory=True):
        """

        :param distance: the method of measuring two vectors, support one of these `euclidean`, `cosine`
        :param save_memory: use for loop method to save memory, but it may be slow in some situations.
        """
        if distance not in self.supported_distance:
            raise NotImplementedError('The distance `{}` is not supported yet, please implement it manually ...'.format(
                distance
            ))
        func = {'euclidean': self.__euclidean, 'cosine': self.__cosine}
        self.distance = func[distance]
        self.save_memory = save_memory

    def __call__(self, x, y, mode):
        """
        compute the distance between x and y. if set mode = 0, return len(y) indexes of x that are most similar to y.
        if set mode =1, return len(x) indexes of y that are most similar to x.

        :param x: whole dataset, the shape likes [samples, dimensions]
        :param y: center vectors, the shape likes [center_samples, dimensions]
        :param mode: set 0 to get x's index, set 1 to get y's index
        :return:
        """
        if self.save_memory:
            return self.__cal(x, y, self.distance, mode)
        else:
            x = x[:, None]
            y = y[None]
            return self.distance(x, y, dim=2).argmin(mode)

    @staticmethod
    def __cal(x, y, distance_fn, mode):
        if mode == 1:
            r = torch.randperm(len(x))
            for index, d in enumerate(x):
                m = distance_fn(d, y, dim=1).argmin(0)
                r[index] = m
        elif mode == 0:
            r = torch.randperm(len(y))
            for index, d in enumerate(y):
                m = distance_fn(d, x, dim=1).argmin(0)
                r[index] = m
        return r

    @staticmethod
    def __euclidean(x, y, dim):
        x = x.sub(y)
        return x.square_().sum(dim)

    @staticmethod
    def __cosine(x, y, dim):
        return torch.cosine_similarity(x, y, dim=dim)