from scipy.sparse import csr_matrix, hstack
import numpy as np

import baseline
import torch
from baseline.processor import get_pred, _get_shape, _fill_tensor


class Processor(baseline.Processor):
    def __init__(self, **kwargs):
        self.keys.add('eval_context_word_idxs')
        self.keys.add('eval_context_glove_idxs')
        self.keys.add('eval_context_char_idxs')
        self.depths['eval_context_word_idxs'] = 2
        self.depths['eval_context_glove_idxs'] = 2
        self.depths['eval_context_char_idxs'] = 3
        super(Processor, self).__init__(**kwargs)

    def postprocess_context(self, example, context_outputs):
        print(example.keys())
        exit()
        # Iterate (eval_len + 1)
        for context_output in context_outputs:
            pos_tuple, dense, sparse_ = context_output
            out = dense.cpu().numpy()
            # TODO: get eval_context and iterate them, too (with idx)
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

    # Override to process 'eval_context'
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
            prepro_example['answer_word_starts'] = answer_word_starts
            prepro_example['answer_word_ends'] = answer_word_ends

        # Added for additional evaluation contexts
        if 'eval_context' in example:
            prepro_example['eval_context_spans'] = []
            prepro_example['eval_context_word_idxs'] = []
            prepro_example['eval_context_glove_idxs'] = []
            prepro_example['eval_context_char_idxs'] = []

            for new_context in example['eval_context']:
                new_context_spans = self._word_tokenize(new_context)
                new_context_words = tuple(new_context[span[0]:span[1]] for span in new_context_spans)
                new_context_word_idxs = tuple(map(self._word2idx, new_context_words))
                new_context_glove_idxs = tuple(map(self._word2idx_ext, new_context_words))
                new_context_char_idxs = tuple(tuple(map(self._char2idx, word)) for word in new_context_words)
                prepro_example['eval_context_spans'].append(new_context_spans)
                prepro_example['eval_context_word_idxs'].append(new_context_word_idxs)
                prepro_example['eval_context_glove_idxs'].append(new_context_glove_idxs)
                prepro_example['eval_context_char_idxs'].append(new_context_char_idxs)

        output = dict(tuple(example.items()) + tuple(prepro_example.items()))

        return output

    # Override to process 'eval_context'
    def collate(self, examples):
        tensors = {}
        for key in self.keys:
            if key not in examples[0]:
                continue
            val = tuple(example[key] for example in examples)
            depth = self.depths[key] + 1
            shape = _get_shape(val, depth)
            tensor = torch.zeros(shape, dtype=torch.int64)
            _fill_tensor(tensor, val)
            tensors[key] = tensor
        if self._elmo:
            if 'context' in examples[0]:
                sentences = [[example['context'][span[0]:span[1]] for span in example['context_spans']]
                             for example in examples]
                character_ids = self._batch_to_ids(sentences)
                tensors['context_elmo_idxs'] = character_ids
            if 'question' in examples[0]:
                sentences = [[example['question'][span[0]:span[1]] for span in example['question_spans']]
                             for example in examples]
                character_ids = self._batch_to_ids(sentences)
                tensors['question_elmo_idxs'] = character_ids

            # Added
            if 'eval_context' in examples[0]:
                sentences = [[[ex[sp[0]:sp[1]] for sp in span]
                    for (ex, span) in zip(example['eval_context'],
                                          example['eval_context_spans'])]
                    for example in examples]
                character_ids = [self._batch_to_ids(sent) for sent in sentences]
                character_ids = torch.stack(character_ids)
                tensors['eval_context_elmo_idxs'] = character_ids
        return tensors


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
