"""Official merge script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys
import shutil

import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import numpy.linalg

from scipy.sparse import csr_matrix, hstack, vstack, save_npz


# Copied from ../baseline/file_interface
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


# Append document tfidf vector to each paragraphs
def append_tfidf(context_emb_dir, question_emb_dir, progress, **kwargs):
    context_emb_dir, context_emb_ext = os.path.splitext(context_emb_dir)
    question_emb_dir, question_emb_ext = os.path.splitext(question_emb_dir)
    if context_emb_ext == '.zip':
        print('Extracting %s to %s' % (context_emb_path, context_emb_dir))
        shutil.unpack_archive(context_emb_path, context_emb_dir)
    if question_emb_ext == '.zip':
        print('Extracting %s to %s' % (question_emb_path, question_emb_dir))
        shutil.unpack_archive(question_emb_path, question_emb_dir)

    if progress:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x

    # Get paths
    context_paths = tuple(os.path.join(context_emb_dir, path)
                          for path in os.listdir(context_emb_dir)
                          if os.path.splitext(path)[1] == '.npz' and \
                          'tfidf' not in path)
    question_paths = tuple(os.path.join(question_emb_dir, path)
                          for path in os.listdir(question_emb_dir)
                          if os.path.splitext(path)[1] == '.npz' and \
                          'tfidf' not in path)
    print('Number of contexts to process: {}'.format(len(context_paths)))
    print('Number of questions to process: {}'.format(len(question_paths)))

    # Load retriever
    from drqa import retriever
    ranker = retriever.get_class('tfidf')(tfidf_path=kwargs['retriever_path'],
                                          strict=False)
    print('Retriever loaded from {}'.format(args.retriever_path))

    # Read SQuAD to construct squad-docs, squad-ques
    squad = _load_squad(args.squad_path)
    assert len(squad) > 0
    squad_docs = {}
    squad_ques = {}
    for item in squad:
        title = '_'.join(item['cid'].split('_')[:-1])
        if title not in squad_docs:
            squad_docs[title] = list()
        if item['context'] not in squad_docs[title]:
            squad_docs[title].append(item['context'])
        squad_ques[item['id']] = item['question']

    # Process contexts first
    tfidf_weight = 1e+2
    for c_emb_path in tqdm(context_paths):

        # Dense vector supported only 
        assert os.path.exists(c_emb_path)
        par_title = os.path.splitext(os.path.basename(c_emb_path))[0]
        doc_title = '_'.join(par_title.split('_')[:-1])
        assert doc_title in squad_docs.keys()
        c_emb = np.load(c_emb_path)  # shape = [N, d]
        c_emb = c_emb['arr_0']

        # Get tf-idf vector of the document, then concat
        tfidf_emb = ranker.text2spvec(' '.join(squad_docs[doc_title]))
        tfidf_emb = tfidf_emb * tfidf_weight         
        tfidf_emb = vstack([tfidf_emb] * c_emb.shape[0]) # tile
        c_emb = csr_matrix(c_emb)
        concat_emb = hstack([c_emb, tfidf_emb])

        # Save concatenated vector
        concat_path = os.path.join(context_emb_dir,
            par_title + '_tfidf{}.npz'.format(tfidf_weight))
        save_npz(concat_path, concat_emb)

    # Process questions
    for q_emb_path in tqdm(question_paths):
        
        # Dense vector supported only
        assert os.path.exists(q_emb_path)
        q_id = os.path.splitext(os.path.basename(q_emb_path))[0]
        assert q_id in squad_ques
        q_emb = np.load(q_emb_path)  # shape = [N, d]
        q_emb = q_emb['arr_0']

        # Get tf-idf vector of question, then concat
        tfidf_emb = ranker.text2spvec(squad_ques[q_id])
        tfidf_emb = tfidf_emb * tfidf_weight
        tfidf_emb = vstack([tfidf_emb] * q_emb.shape[0]) # tile
        q_emb = csr_matrix(q_emb)
        concat_emb = hstack([q_emb, tfidf_emb])

        # Save concatenated vector
        concat_path = os.path.join(question_emb_dir,
            q_id + '_tfidf{}.npz'.format(tfidf_weight))
        save_npz(concat_path, concat_emb)


# Predefined paths
home = os.path.expanduser('~')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
SQUAD_PATH = os.path.join(home, 'data/squad',
    'dev-v1.1.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for appending tf-idf')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--squad-path', type=str, default=SQUAD_PATH,
                        help='SQuAD dataset path')
    parser.add_argument('--progress', default=False, action='store_true')
    args = parser.parse_args()

    append_tfidf(**args.__dict__)

