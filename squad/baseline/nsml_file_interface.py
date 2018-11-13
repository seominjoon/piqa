import json
import os

import numpy as np

import base.nsml_file_interface


class FileInterface(base.nsml_file_interface.FileInterface):
    def __init__(self, glove_dir, glove_size, elmo_options_file, elmo_weights_file, **kwargs):
        glove_dir = '/static/glove_squad'
        elmo_options_file = '/static/elmo/options.json'
        elmo_weights_file = '/static/elmo/weights.hdf5'
        self._glove_dir = glove_dir
        self._glove_size = glove_size
        self._elmo_options_file = elmo_options_file
        self._elmo_weights_file = elmo_weights_file
        super(FileInterface, self).__init__(**kwargs)

    def load_train(self):
        return _load_squad(self._train_path, draft=self._draft)

    def load_test(self):
        return _load_squad(self._test_path, draft=self._draft)

    def load_metadata(self):
        glove_vocab, glove_emb_mat = _load_glove(self._glove_size, glove_dir=self._glove_dir, draft=self._draft)
        return {'glove_vocab': glove_vocab,
                'glove_emb_mat': glove_emb_mat,
                'elmo_options_file': self._elmo_options_file,
                'elmo_weights_file': self._elmo_weights_file}


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
    with open(glove_path, 'rb') as fp:
        vocab = []
        vecs = []
        for idx, line in enumerate(fp):
            line = line.decode('utf-8')
            tokens = line.strip().split(u' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            vecs.append(vec)
            vocab.append(word)
            if draft and idx >= 99:
                break
    emb_mat = np.array(vecs, dtype=np.float32)
    return vocab, emb_mat

