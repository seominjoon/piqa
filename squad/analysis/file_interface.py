import dev
import json
import os
import numpy as np
import scipy
import random


class FileInterface(dev.FileInterface):
    def __init__(self, **kwargs):
        super(FileInterface, self).__init__(**kwargs)

    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)
        return squad
