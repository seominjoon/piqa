import random

from scipy.sparse import csr_matrix, hstack
import numpy as np

import baseline
import base
from baseline.processor import get_pred


class Processor(baseline.Processor):
    def postprocess_context(self, example, context_output):
        pos_tuple, dense, sparse_ = context_output
        out = dense.cpu().numpy()
        context = example['context']
        context_spans = example['context_spans']
        phrases = tuple(get_pred(context, context_spans, yp1, yp2) for yp1, yp2 in pos_tuple)
        if self._emb_type == 'sparse' or sparse_ is not None:
            out = csr_matrix(out)
            if sparse_ is not None:
                idx, val, max_ = sparse_
                sparse_tensor = SparseTensor(idx.cpu().numpy(), val.cpu().numpy(), max_)
                out = hstack([out, sparse_tensor.scipy()])
        metadata = {'context': context,
                    'answer_spans': tuple((context_spans[yp1][0], context_spans[yp2][1]) for yp1, yp2 in pos_tuple)}
        return example['cid'], phrases, out, metadata

    def postprocess_question(self, example, question_output):
        dense, sparse = question_output
        out = dense.cpu().numpy()
        if self._emb_type == 'sparse' or sparse is not None:
            out = csr_matrix(out)
            if sparse is not None:
                idx, val, max_ = sparse
                sparse_tensor = SparseTensor(idx.cpu().numpy(), val.cpu().numpy(), max_)
                out = hstack([out, sparse_tensor.scipy()])
        return example['id'], out


class Sampler(base.Sampler):
    def __init__(self, dataset, data_type, max_context_size=None, max_question_size=None, bucket=False, shuffle=False,
                 max_ans_len=None, **kwargs):
        super(Sampler, self).__init__(dataset, data_type)
        if data_type == 'dev' or data_type == 'test':
            max_context_size = None
            max_question_size = None
            max_ans_len = None
            self.shuffle = False

        self.max_context_size = max_context_size
        self.max_question_size = max_question_size
        self.shuffle = shuffle
        self.max_ans_len = max_ans_len
        self.bucket = bucket

        idxs = tuple(idx for idx in range(len(dataset))
                     if (max_context_size is None or len(dataset[idx]['context_spans']) <= max_context_size) and
                     (max_question_size is None or len(dataset[idx]['question_spans']) <= max_question_size) and
                     (max_ans_len is None or dataset[idx]['word_answer_ends'][0] - dataset[idx]['word_answer_starts'][
                         0] < max_ans_len))
        print('%s: using %d/%d examples' % (data_type, len(idxs), len(dataset)))

        if shuffle:
            idxs = random.sample(idxs, len(idxs))

        if bucket:
            if 'context_spans' in dataset[0]:
                idxs = sorted(idxs, key=lambda idx: len(dataset[idx]['context_spans']))
            else:
                assert 'question_spans' in dataset[0]
                idxs = sorted(idxs, key=lambda idx: len(dataset[idx]['question_spans']))
        self._idxs = idxs

    def __iter__(self):
        return iter(self._idxs)

    def __len__(self):
        return len(self._idxs)


class SparseTensor(object):
    def __init__(self, idx, val, max_=None):
        self.idx = idx
        self.val = val
        self.max = max_

    def scipy(self):
        col = self.idx.flatten()
        row = np.tile(np.expand_dims(range(self.idx.shape[0]), 1), [1, self.idx.shape[1]]).flatten()
        data = self.val.flatten()
        shape = None if self.max is None else [self.idx.shape[0], self.max]
        return csr_matrix((data, (row, col)), shape=shape)
