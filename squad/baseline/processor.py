import random
import re
import string
from collections import Counter

import nltk
import torch
import torch.utils.data
from scipy.sparse import csc_matrix
import numpy as np

import base


class Tokenizer(object):
    def tokenize(self, in_):
        raise NotImplementedError()


class PTBSentTokenizer(Tokenizer):
    def tokenize(self, in_):
        sents = nltk.sent_tokenize(in_)
        return _get_spans(in_, sents)


class PTBWordTokenizer(Tokenizer):
    def tokenize(self, in_):
        in_ = in_.replace('``', '" ').replace("''", '" ').replace('\t', ' ')
        words = nltk.word_tokenize(in_)
        words = tuple(word.replace('``', '"').replace("''", '"') for word in words)
        return _get_spans(in_, words)


class Processor(base.Processor):
    keys = {'context_word_idxs',
            'context_glove_idxs',
            'context_char_idxs',
            'question_word_idxs',
            'question_glove_idxs',
            'question_char_idxs',
            'answer_word_starts',
            'answer_word_ends',
            'idx'}
    depths = {'context_word_idxs': 1,
              'context_glove_idxs': 1,
              'context_char_idxs': 2,
              'question_word_idxs': 1,
              'question_glove_idxs': 1,
              'question_char_idxs': 2,
              'answer_word_starts': 1,
              'answer_word_ends': 1,
              'idx': 0}
    pad = '<pad>'
    unk = '<unk>'

    def __init__(self, char_vocab_size=None, glove_vocab_size=None, word_vocab_size=None, elmo=False, draft=False,
                 emb_type=None, **kwargs):
        self._word_tokenizer = PTBWordTokenizer()
        self._sent_tokenizer = PTBSentTokenizer()
        self._char_vocab_size = char_vocab_size
        self._glove_vocab_size = glove_vocab_size
        self._word_vocab_size = word_vocab_size
        self._elmo = elmo
        if elmo:
            from allennlp.modules.elmo import batch_to_ids
            self._batch_to_ids = batch_to_ids
        self._draft = draft
        self._emb_type = emb_type
        self._glove = None

        self._word_cache = {}
        self._sent_cache = {}
        self._word2idx_dict = {}
        self._word2idx_ext_dict = {}
        self._char2idx_dict = {}

    def construct(self, examples, metadata):
        assert metadata is not None
        glove_vocab = metadata['glove_vocab']
        word_counter, lower_word_counter, char_counter = Counter(), Counter(), Counter()
        for example in examples:
            for text in (example['context'], example['question']):
                for span in self._word_tokenize(example['context']):
                    word = text[span[0]:span[1]]
                    word_counter[word] += 1
                    lower_word_counter[word] += 1
                    for char in word:
                        char_counter[char] += 1

        word_vocab = tuple(item[0] for item in sorted(word_counter.items(), key=lambda item: -item[1]))
        word_vocab = (Processor.pad, Processor.unk) + word_vocab
        word_vocab = word_vocab[:self._word_vocab_size] if len(word_vocab) > self._word_vocab_size else word_vocab
        self._word2idx_dict = {word: idx for idx, word in enumerate(word_vocab)}

        char_vocab = tuple(item[0] for item in sorted(char_counter.items(), key=lambda item: -item[1]))
        char_vocab = (Processor.pad, Processor.unk) + char_vocab
        char_vocab = char_vocab[:self._char_vocab_size] if len(char_vocab) > self._char_vocab_size else char_vocab
        self._char2idx_dict = {char: idx for idx, char in enumerate(char_vocab)}

        ext_vocab = (Processor.pad, Processor.unk) + tuple(glove_vocab)
        if len(ext_vocab) > self._glove_vocab_size:
            ext_vocab = ext_vocab[:self._glove_vocab_size]
        self._word2idx_ext_dict = {ext: idx for idx, ext in enumerate(ext_vocab)}
        # assert max(self._word2idx_ext.values()) + 1 == self._glove_vocab_size, max(self._word2idx_ext.values()) + 1

    def state_dict(self):
        out = {'word2idx': self._word2idx,
               'word2idx_ext': self._word2idx_ext,
               'char2idx': self._char2idx}
        return out

    def load_state_dict(self, in_):
        self._word2idx_dict = in_['word2idx']
        self._word2idx_ext_dict = in_['word2idx_ext']
        self._char2idx_dict = in_['char2idx']

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

        output = dict(tuple(example.items()) + tuple(prepro_example.items()))
        return output

    def postprocess(self, example, model_output):
        yp1 = model_output['yp1'].item()
        yp2 = model_output['yp2'].item()
        context = example['context']
        context_spans = example['context_spans']
        pred = _get_pred(context, context_spans, yp1, yp2)
        out = {'pred': pred, 'id': example['id']}
        if 'answer_starts' in example:
            y1 = example['answer_starts']
            y2 = example['answer_ends']
            gt = [context[s:e] for s, e in zip(y1, y2)]
            f1 = max(_f1_score(pred, gt_each) for gt_each in gt)
            em = max(_exact_match_score(pred, gt_each) for gt_each in gt)
            out['gt'] = gt
            out['f1'] = f1
            out['em'] = em
        return out

    def postprocess_batch(self, dataset, model_input, model_output):
        results = tuple(self.postprocess(dataset[idx],
                                         {key: val[i] if val is not None else None for key, val in
                                          model_output.items()})
                        for i, idx in enumerate(model_input['idx']))
        return results

    def postprocess_context(self, example, context_output):
        pos_tuple, dense = context_output
        out = dense.cpu().numpy()
        context = example['context']
        context_spans = example['context_spans']
        phrases = tuple(_get_pred(context, context_spans, yp1, yp2) for yp1, yp2 in pos_tuple)
        if self._emb_type == 'sparse':
            out = csc_matrix(out)
        return example['cid'], phrases, out

    def postprocess_context_batch(self, dataset, model_input, context_output):
        results = tuple(self.postprocess_context(dataset[idx], context_output[i])
                        for i, idx in enumerate(model_input['idx']))
        return results

    def postprocess_question(self, example, question_output):
        dense = question_output
        out = dense.cpu().numpy()
        if self._emb_type == 'sparse':
            out = csc_matrix(out)
        return example['id'], out

    def postprocess_question_batch(self, dataset, model_input, question_output):
        results = tuple(self.postprocess_question(dataset[idx], question_output[i])
                        for i, idx in enumerate(model_input['idx']))
        return results

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
        return tensors

    def process_metadata(self, metadata):
        return {'glove_emb_mat': torch.tensor(metadata['glove_emb_mat'])}

    def get_dump(self, dataset, input_, output, results):
        dump = []
        for i, idx in enumerate(input_['idx']):
            example = dataset[idx]
            each = {'id': example['id'],
                    'context': example['context'],
                    'question': example['question'],
                    'answer_starts': example['answer_starts'],
                    'answer_ends': example['answer_ends'],
                    'context_spans': example['context_spans'],
                    'yp1': output['yp1'][i].cpu().numpy(),
                    'yp2': output['yp2'][i].cpu().numpy(),
                    }
            dump.append(each)
        return dump

    # private methods below
    def _word_tokenize(self, string):
        if string in self._word_cache:
            return self._word_cache[string]
        spans = self._word_tokenizer.tokenize(string)
        self._word_cache[string] = spans
        return spans

    def _sent_tokenize(self, string):
        if string in self._sent_cache:
            return self._sent_cache[string]
        spans = self._sent_tokenizer.tokenize(string)
        self._sent_cache[string] = spans
        return spans

    def _word2idx(self, word):
        return self._word2idx_dict[word] if word in self._word2idx_dict else 1

    def _word2idx_ext(self, word):
        word = word.lower()
        return self._word2idx_ext_dict[word] if word in self._word2idx_ext_dict else 1

    def _char2idx(self, char):
        return self._char2idx_dict[char] if char in self._char2idx_dict else 1


class Sampler(base.Sampler):
    def __init__(self, dataset, data_type, max_context_size=None, max_question_size=None, bucket=False, shuffle=False,
                 **kwargs):
        super(Sampler, self).__init__(dataset, data_type)
        if data_type == 'dev' or data_type == 'test':
            max_context_size = None
            max_question_size = None
            self.shuffle = False

        self.max_context_size = max_context_size
        self.max_question_size = max_question_size
        self.shuffle = shuffle
        self.bucket = bucket

        idxs = tuple(idx for idx in range(len(dataset))
                     if (max_context_size is None or len(dataset[idx]['context_spans']) <= max_context_size) and
                     (max_question_size is None or len(dataset[idx]['question_spans']) <= max_question_size))

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
        return csc_matrix((data, (row, col)), shape=shape)


# SquadProcessor-specific helpers

def _get_pred(context, spans, yp1, yp2):
    if yp1 >= len(spans):
        print('warning: yp1 is set to 0')
        yp1 = 0
    if yp2 >= len(spans):
        print('warning: yp1 is set to 0')
        yp2 = 0
    yp1c = spans[yp1][0]
    yp2c = spans[yp2][1]
    return context[yp1c:yp2c]


def _get_spans(in_, tokens):
    pairs = []
    i = 0
    for token in tokens:
        i = in_.find(token, i)
        assert i >= 0, 'token `%s` not found starting from %d: `%s`' % (token, i, in_[i:])
        pair = (i, i + len(token))
        pairs.append(pair)
        i += len(token)
    return tuple(pairs)


def _get_shape(nested_list, depth):
    if depth > 0:
        return (len(nested_list),) + tuple(map(max, zip(*[_get_shape(each, depth - 1) for each in nested_list])))
    return ()


def _fill_tensor(tensor, nested_list):
    if tensor.dim() == 1:
        tensor[:len(nested_list)] = torch.tensor(nested_list)
    elif tensor.dim() == 2:
        for i, each in enumerate(nested_list):
            tensor[i, :len(each)] = torch.tensor(each)
    elif tensor.dim() == 3:
        for i1, each1 in enumerate(nested_list):
            for i2, each2 in enumerate(each1):
                tensor[i1, i2, :len(each2)] = torch.tensor(each2)
    else:
        for tensor_child, nested_list_child in zip(tensor, nested_list):
            _fill_tensor(tensor_child, nested_list_child)


# SQuAD official evaluation helpers

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.

    Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED.

    Args:
      s: Input text.
    Returns:
      Normalized text.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction, ground_truth):
    """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _exact_match_score(prediction, ground_truth):
    """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)
