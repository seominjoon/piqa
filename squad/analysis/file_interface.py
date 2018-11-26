import dev
import json
import os
import numpy as np
import scipy
import random


class FileInterface(dev.FileInterface):
    def __init__(self, **kwargs):
        self.analysis = kwargs['analysis']
        super(FileInterface, self).__init__(**kwargs)

    def load_test(self):
        with open(self._test_path, 'r') as fp:
            squad = json.load(fp)

        if self.analysis == 'eval':
            return squad
        elif self.analysis == 'open':
            test_examples = [{"idx": int("{}".format(idx)),
                              "cid": "Open_{}".format(idx),
                              "context": context} for (idx, context) in
                              enumerate(squad)]
            return test_examples
        else:
            return super().load_test()

    def question_load(self, emb_type='dense', shuffle=True, tfidf=False):
        paths = os.listdir(self._question_emb_dir)
        if shuffle:
            random.shuffle(paths)
        if not tfidf:
            npz_paths = tuple('%s.npz' % os.path.join(self._question_emb_dir, os.path.splitext(path)[0]) for path in paths if 'tfidf' not in path)
        else:
            npz_paths = tuple('%s_tfidf.npz' % os.path.join(self._question_emb_dir, os.path.splitext(path)[0]) for path in paths)
        embs = []
        for npz_path in npz_paths:
            if emb_type == 'dense' and not tfidf:
                emb = np.load(npz_path)['arr_0']
            else:
                emb = scipy.sparse.load_npz(npz_path)

            # squeezed
            embs.append(emb[0])

        ids = [os.path.splitext(os.path.basename(path))[0] for path in npz_paths]
        return embs, ids

    def context_load(self, metadata=False, emb_type='dense', shuffle=True, tfidf=False):
        paths = os.listdir(self._context_emb_dir)
        if shuffle:
            random.shuffle(paths)
        json_paths = tuple(os.path.join(self._context_emb_dir, path)
                           for path in paths if os.path.splitext(path)[1] == '.json')
        if not tfidf:
            npz_paths = tuple('%s.npz' % os.path.splitext(path)[0] for path in json_paths)
        else:
            npz_paths = tuple('%s_tfidf.npz' % os.path.splitext(path)[0] for path in json_paths)

        metadata_paths = tuple('%s.metadata' % os.path.splitext(path)[0] for path in json_paths)
        for json_path, npz_path, metadata_path in zip(json_paths, npz_paths, metadata_paths):
            with open(json_path, 'r') as fp:
                phrases = json.load(fp)
            if emb_type == 'dense' and not tfidf:
                emb = np.load(npz_path)['arr_0']
            else:
                emb = scipy.sparse.load_npz(npz_path)
            if metadata:
                with open(metadata_path, 'r') as fp:
                    metadata = json.load(fp)
                yield phrases, emb, metadata
            else:
                yield phrases, emb
