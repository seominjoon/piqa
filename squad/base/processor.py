from abc import ABCMeta

import torch.utils.data


class Processor(metaclass=ABCMeta):
    pass


class Sampler(torch.utils.data.Sampler, metaclass=ABCMeta):
    pass
