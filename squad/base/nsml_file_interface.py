import os

import torch
import nsml

import base.file_interface


class FileInterface(base.file_interface.FileInterface):
    def __init__(self, draft, **kwargs):
        train_path = os.path.join(nsml.DATASET_PATH, 'train', 'train-v1.1.json')
        test_path = os.path.join(nsml.DATASET_PATH, 'train', 'dev-v1.1.json')
        save_dir = None
        report_path = None
        dump_dir = './dump/'
        pred_path = './pred.json'
        question_emb_dir = './question_emb/'
        context_emb_dir = './context_emb/'
        cache_path = './preprocess.pt'

        glove_dir = '/static/glove_squad'
        # elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        # elmo_weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        elmo_options_file = '/static/elmo/options.json'
        elmo_weights_file = '/static/elmo/weights.hdf5'

        super(FileInterface, self).__init__(save_dir,
                                            report_path,
                                            pred_path,
                                            question_emb_dir,
                                            context_emb_dir,
                                            cache_path,
                                            dump_dir,
                                            train_path,
                                            test_path,
                                            draft)

    def _bind(self, save=None, load=None, infer=None):
        nsml.bind(save=save, load=load, infer=infer)

    def save(self, iteration, save_fn=None):
        nsml.save(iteration, save_fn=save_fn)

    def load(self, iteration, load_fn=None, session=None):
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
