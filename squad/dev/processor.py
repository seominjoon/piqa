from scipy.sparse import csr_matrix, hstack
import numpy as np

import baseline
from baseline.processor import get_pred, _f1_score, _exact_match_score


class Processor(baseline.Processor):
    def preprocess(self, example):
        prepro_example = {'idx': example['idx']}

        if 'context' in example:
            context = example['context']
            context_spans = self._word_tokenize(context)
            context_words = tuple(context[span[0]:span[1]] for span in context_spans)
            context_word_idxs = tuple(map(self._word2idx, context_words))
            context_glove_idxs = tuple(map(self._word2idx_ext, context_words))
            context_char_idxs = tuple(tuple(map(self._char2idx, word)) for word in context_words)
            prepro_example['context_spans'] = context_spans
            prepro_example['context_word_idxs'] = context_word_idxs
            prepro_example['context_glove_idxs'] = context_glove_idxs
            prepro_example['context_char_idxs'] = context_char_idxs

        if 'question' in example:
            question = example['question']
            question_spans = self._word_tokenize(example['question'])
            question_words = tuple(question[span[0]:span[1]] for span in question_spans)
            question_word_idxs = tuple(map(self._word2idx, question_words))
            question_glove_idxs = tuple(map(self._word2idx_ext, question_words))
            question_char_idxs = tuple(tuple(map(self._char2idx, word)) for word in question_words)
            prepro_example['question_spans'] = question_spans
            prepro_example['question_word_idxs'] = question_word_idxs
            prepro_example['question_glove_idxs'] = question_glove_idxs
            prepro_example['question_char_idxs'] = question_char_idxs

        if 'answer_starts' in example:
            answer_word_start, answer_word_end = 0, 0
            answer_word_starts, answer_word_ends = [], []
            for answer_start in example['answer_starts']:
                for word_idx, span in enumerate(context_spans):
                    if span[0] <= answer_start:
                        answer_word_start = word_idx + 1
                answer_word_starts.append(answer_word_start)
            for answer_end in example['answer_ends']:
                for word_idx, span in enumerate(context_spans):
                    if span[0] <= answer_end:
                        answer_word_end = word_idx + 1
                answer_word_ends.append(answer_word_end)

            # no answer capa
            if len(answer_word_starts) == 0:
                assert len(answer_word_ends) == 0
                answer_word_starts.append(0)
                answer_word_ends.append(0)

            prepro_example['answer_word_starts'] = answer_word_starts
            prepro_example['answer_word_ends'] = answer_word_ends

        output = dict(tuple(example.items()) + tuple(prepro_example.items()))
        return output

    def postprocess(self, example, model_output):
        yp1 = model_output['yp1'].item()
        yp2 = model_output['yp2'].item()
        context = example['context']
        context_spans = example['context_spans']
        pred = get_pred(context, context_spans, yp1, yp2)
        out = {'pred': pred, 'id': example['id']}
        if 'answer_starts' in example:
            y1 = example['answer_starts']
            y2 = example['answer_ends']
            gt = [context[s:e] for s, e in zip(y1, y2)]
            out['gt'] = gt

            if len(gt) > 0:
                f1 = max(_f1_score(pred, gt_each) for gt_each in gt)
                em = max(_exact_match_score(pred, gt_each) for gt_each in gt)
                out['f1'] = f1
                out['em'] = em
        return out

    def postprocess_context(self, example, context_output):
        pos_tuple, dense, sparse_, fsp = context_output
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
        if fsp is None:
            probs = None
        else:
            probs = [round(a, 4) for a in fsp.tolist()]
        metadata = {'context': context,
                    'answer_spans': tuple((context_spans[yp1][0], context_spans[yp2][1]) for yp1, yp2 in pos_tuple),
                    'probs': probs}
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
