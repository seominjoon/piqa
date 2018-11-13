import os

import torch
import nsml

import base.file_interface


class FileInterface(base.file_interface.FileInterface):
    def __init__(self, cuda, mode, save_dir, load_dir, report_path, pred_path, question_emb_dir, context_emb_dir,
                 cache_path, dump_dir, train_path, test_path, draft, **kwargs):
        train_path = os.path.join(nsml.DATASET_PATH, 'train', 'train-v1.1.json')
        test_path = os.path.join(nsml.DATASET_PATH, 'train', 'dev-v1.1.json')
        save_dir = './save/'
        report_path = './report.csv'
        dump_dir = './dump/'
        pred_path = './pred.json'
        question_emb_dir = './question_emb/'
        context_emb_dir = './context_emb/'
        cache_path = './preprocess.pt'

        super(FileInterface, self).__init__(cuda, mode, save_dir, load_dir, report_path, pred_path, question_emb_dir,
                                            context_emb_dir, cache_path, dump_dir, train_path, test_path, draft,
                                            **kwargs)

    def _bind(self, save=None, load=None, infer=None):
        nsml.bind(save=save, load=load, infer=infer)

    def save(self, iteration, save_fn=None):
        nsml.save(iteration, save_fn=save_fn)

    def load(self, iteration='0', load_fn=None, session=None):
        nsml.load(iteration, load_fn=load_fn, session=session)

    def report(self, **kwargs):
        nsml.report(**kwargs)
        return ', '.join('%s=%.5r' % (s, r) for s, r in kwargs.items())

    def cache(self, preprocess, args, **kwargs):
        # nsml always saves

        def p(output_path, data, **local):
            torch.save(preprocess(self, args, **local), output_path[0])

        nsml.cache(preprocess_fn=p, output_path=[self._cache_path], data=None, **kwargs)

        return torch.load(self._cache_path)
