from scipy.sparse import csr_matrix, hstack
import numpy as np

import baseline
from baseline.processor import get_pred


class Processor(baseline.Processor):
    def postprocess_context(self, example, context_output):
        pos_tuple, dense, sparse_ = context_output
        out = dense.cpu().numpy()
        context = example['context']
        context_spans = example['context_spans']
        phrases = tuple(get_pred(context, context_spans, yp1, yp2) for yp1, yp2 in pos_tuple)
        if self._emb_type == 'sparse' or sparse_ is not None:
            out = csc_matrix(out)
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


class Sampler(baseline.Sampler):
    pass


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
