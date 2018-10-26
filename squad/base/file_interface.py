import json
import os
import shutil

import torch
import scipy.sparse
import numpy as np
import csv


class FileInterface(object):
    def __init__(self, cuda, mode, save_dir, report_path, pred_path, question_emb_dir, context_emb_dir,
                 cache_path, dump_dir, train_path, test_path, draft, **kwargs):
        self._cuda = cuda
        self._mode = mode
        self._train_path = train_path
        self._test_path = test_path
        self._save_dir = save_dir
        self._report_path = report_path
        self._dump_dir = dump_dir
        self._pred_path = pred_path
        self._question_emb_dir = os.path.splitext(question_emb_dir)[0]
        self._context_emb_dir = os.path.splitext(context_emb_dir)[0]
        self._cache_path = cache_path
        self._args_path = os.path.join(save_dir, 'args.json')
        self._draft = draft
        self._save = None
        self._load = None
        self._report_header = []
        self._report = []
        self._kwargs = kwargs

    def _bind(self, save=None, load=None):
        self._save = save
        self._load = load

    def save(self, iteration, save_fn=None):
        filename = os.path.join(self._save_dir, str(iteration))
        if not os.path.exists(filename):
            os.makedirs(filename)
        if save_fn is None:
            save_fn = self._save
        save_fn(filename)

    def save_args(self, args):
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        with open(self._args_path, 'w') as fp:
            json.dump(args, fp)

    def load(self, iteration, load_fn=None, session=None):
        if session is None:
            session = self._save_dir
        filename = os.path.join(session, str(iteration), 'model.pt')
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if load_fn is None:
            load_fn = self._load
        load_fn(filename)

    def pred(self, pred):
        if not os.path.exists(os.path.dirname(self._pred_path)):
            os.makedirs(os.path.dirname(self._pred_path))
        with open(self._pred_path, 'w') as fp:
            json.dump(pred, fp)
            print('Prediction saved at %s' % self._pred_path)

    def report(self, summary=False, **kwargs):
        if not os.path.exists(os.path.dirname(self._report_path)):
            os.makedirs(os.path.dirname(self._report_path))
        if len(self._report) == 0 and os.path.exists(self._report_path):
            with open(self._report_path, 'r') as fp:
                reader = csv.DictReader(fp, delimiter=',')
                rows = list(reader)
                for key in rows[0]:
                    if key not in self._report_header:
                        self._report_header.append(key)
                self._report.extend(rows)

        for key, val in kwargs.items():
            if key not in self._report_header:
                self._report_header.append(key)
        self._report.append(kwargs)
        with open(self._report_path, 'w') as fp:
            writer = csv.DictWriter(fp, delimiter=',', fieldnames=self._report_header)
            writer.writeheader()
            writer.writerows(self._report)
        return ', '.join('%s=%.5r' % (s, r) for s, r in kwargs.items())

    def question_emb(self, id_, emb, emb_type='dense'):
        if not os.path.exists(self._question_emb_dir):
            os.makedirs(self._question_emb_dir)
        savez = scipy.sparse.save_npz if emb_type == 'sparse' else np.savez
        path = os.path.join(self._question_emb_dir, '%s.npz' % id_)
        savez(path, emb)

    def context_emb(self, id_, phrases, emb, emb_type='dense'):
        if not os.path.exists(self._context_emb_dir):
            os.makedirs(self._context_emb_dir)
        savez = scipy.sparse.save_npz if emb_type == 'sparse' else np.savez
        emb_path = os.path.join(self._context_emb_dir, '%s.npz' % id_)
        json_path = os.path.join(self._context_emb_dir, '%s.json' % id_)

        if os.path.exists(emb_path):
            print('Skipping %s; already exists' % emb_path)
        else:
            savez(emb_path, emb)
        if os.path.exists(json_path):
            print('Skipping %s; already exists' % json_path)
        else:
            with open(json_path, 'w') as fp:
                json.dump(phrases, fp)

    def archive(self):
        if self._mode == 'embed' or self._mode == 'embed_context':
            shutil.make_archive(self._context_emb_dir, 'zip', self._context_emb_dir)
            shutil.rmtree(self._context_emb_dir)

        if self._mode == 'embed' or self._mode == 'embed_question':
            shutil.make_archive(self._question_emb_dir, 'zip', self._question_emb_dir)
            shutil.rmtree(self._question_emb_dir)

    def cache(self, preprocess, args):
        if os.path.exists(self._cache_path):
            return torch.load(self._cache_path)
        out = preprocess(self, args)
        torch.save(out, self._cache_path)
        return out

    def dump(self, batch_idx, item):
        filename = os.path.join(self._dump_dir, '%s.pt' % str(batch_idx).zfill(6))
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(item, filename)

    def bind(self, processor, model, optimizer=None):
        def load(filename, **kwargs):
            # filename = os.path.join(filename, 'model.pt')
            state = torch.load(filename, map_location=None if self._cuda else 'cpu')
            processor.load_state_dict(state['preprocessor'])
            model.load_state_dict(state['model'])
            if 'optimizer' in state and optimizer:
                optimizer.load_state_dict(state['optimizer'])
            print('Model loaded from %s' % filename)

        def save(filename, **kwargs):
            state = {
                'preprocessor': processor.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            filename = os.path.join(filename, 'model.pt')
            torch.save(state, filename)
            print('Model saved at %s' % filename)

        def infer(input, top_k=100):
            # input = {'id': '', 'question': '', 'context': ''}
            model.eval()

        self._bind(save=save, load=load)

    def load_train(self):
        raise NotImplementedError()

    def load_test(self):
        raise NotImplementedError()

    def load_metadata(self):
        raise NotImplementedError()
