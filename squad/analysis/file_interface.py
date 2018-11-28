import dev
import json
import os
import numpy as np
import scipy
import random


class FileInterface(dev.FileInterface):
    def __init__(self, **kwargs):
        # self.analysis = kwargs['analysis']
        super(FileInterface, self).__init__(**kwargs)

    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)
        return squad

        # Deprecated
        '''
        if self.analysis == 'large':
            return squad
        # Used by open_merge.py
        elif self.analysis == 'open':
            test_examples = [{"idx": int("{}".format(idx)),
                              "cid": "Open_{}".format(idx),
                              "context": context} for (idx, context) in
                              enumerate(squad)]
            return test_examples
        else:
            return super().load_test()
        '''

    # Used by open_merge.py (deprecated)
    def question_load(self, emb_type='dense', shuffle=True):
        paths = os.listdir(self._question_emb_dir)
        if shuffle:
            random.shuffle(paths)
        npz_paths = tuple(
            '%s.npz' % os.path.join(
                self._question_emb_dir, 
                os.path.splitext(path)[0]) 
            for path in paths
        )
        embs = []
        for npz_path in npz_paths:
            if emb_type == 'dense':
                emb = np.load(npz_path)['arr_0']
            else:
                emb = scipy.sparse.load_npz(npz_path)

            # squeezed
            embs.append(emb[0])

        ids = [os.path.splitext(os.path.basename(path))[0] for path in npz_paths]
        return embs, ids

