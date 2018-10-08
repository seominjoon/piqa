from abc import ABCMeta

from torch import nn


class Model(nn.Module, metaclass=ABCMeta):
    def forward(self, *input):
        """

        :param input:
        :return: a dict of tensors
        """
        raise NotImplementedError()

    def get_context(self, *args, **kwargs):
        raise NotImplementedError()

    def get_question(self, *args, **kwargs):
        raise NotImplementedError()


class Loss(nn.Module, metaclass=ABCMeta):
    def forward(self, *input):
        raise NotImplementedError()
