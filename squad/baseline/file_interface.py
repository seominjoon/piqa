import json
import os
import torch

import scipy.sparse
import numpy as np
import csv


class FileInterface(object):
    def __init__(self, save_dir, report_path, pred_path, question_emb_dir, context_emb_dir,
                 cache_path, dump_dir, train_path, test_path, draft, **kwargs):
        self._train_path = train_path
        self._test_path = test_path
        self._save_dir = save_dir
        self._report_path = report_path
        self._dump_dir = dump_dir
        self._pred_path = pred_path
        self._question_emb_dir = question_emb_dir
        self._context_emb_dir = context_emb_dir
        self._cache_path = cache_path
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

    def load_train(self):
        return _load_squad(self._train_path, draft=self._draft)

    def load_test(self):
        return _load_squad(self._test_path, draft=self._draft)

    def load_metadata(self):
        glove_size = self._kwargs['glove_size']
        glove_dir = self._kwargs['glove_dir']
        glove_vocab, glove_emb_mat = _load_glove(glove_size, glove_dir=glove_dir, draft=self._draft)
        return {'glove_vocab': glove_vocab,
                'glove_emb_mat': glove_emb_mat}

    def bind(self, processor, model, optimizer=None):
        def load(filename, **kwargs):
            # filename = os.path.join(filename, 'model.pt')
            state = torch.load(filename)
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


def _load_squad(squad_path, draft=False):
    with open(squad_path, 'r') as fp:
        squad = json.load(fp)
        examples = []
        for article in squad['data']:
            for para_idx, paragraph in enumerate(article['paragraphs']):
                cid = '%s_%d' % (article['title'], para_idx)
                if 'context' in paragraph:
                    context = paragraph['context']
                    context_example = {'cid': cid, 'context': context}
                else:
                    context_example = {}

                if 'qas' in paragraph:
                    for question_idx, qa in enumerate(paragraph['qas']):
                        id_ = qa['id']
                        qid = '%s_%d' % (cid, question_idx)
                        question = qa['question']
                        question_example = {'id': id_, 'qid': qid, 'question': question}
                        if 'answers' in qa:
                            answers, answer_starts, answer_ends = [], [], []
                            for answer in qa['answers']:
                                answer_start = answer['answer_start']
                                answer_end = answer_start + len(answer['text'])
                                answers.append(answer['text'])
                                answer_starts.append(answer_start)
                                answer_ends.append(answer_end)
                            answer_example = {'answers': answers, 'answer_starts': answer_starts,
                                              'answer_ends': answer_ends}
                            question_example.update(answer_example)

                        example = {'idx': len(examples)}
                        example.update(context_example)
                        example.update(question_example)
                        examples.append(example)
                        if draft and len(examples) == 100:
                            return examples
                else:
                    example = {'idx': len(examples)}
                    example.update(context_example)
                    examples.append(example)
                    if draft and len(examples) == 100:
                        return examples
        return examples


def _load_glove(size, glove_dir=None, draft=False):
    if glove_dir is None:
        glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip'
        raise NotImplementedError()

    glove_path = os.path.join(glove_dir, 'glove.6B.%dd.txt' % size)
    with open(glove_path, 'r') as fp:
        vocab = []
        vecs = []
        for idx, line in enumerate(fp):
            # line = line.decode('utf-8')
            tokens = line.strip().split(u' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            vecs.append(vec)
            vocab.append(word)
            if draft and idx >= 99:
                break
    emb_mat = np.array(vecs, dtype=np.float32)
    return vocab, emb_mat

