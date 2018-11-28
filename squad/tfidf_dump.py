"""Official script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys
import numpy as np
import pickle

from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm


# Dump document/question tfidf vector as files (.tfidf)
def dump_tfidf(context_tfidf_dir, question_tfidf_dir, **kwargs):

    # Load retriever
    from drqa import retriever
    ranker = retriever.get_class('tfidf')(
        tfidf_path=kwargs['retriever_path'],
        strict=False
    )
    print('Retriever loaded from {}'.format(kwargs['retriever_path']))

    # Read SQuAD to construct squad-docs, squad-ques
    from analysis.expand_dev import squad_docs_ques
    from baseline.file_interface import _load_squad
    squad = _load_squad(kwargs['squad_path'])
    squad_docs, squad_ques = squad_docs_ques(squad)

    # Get negative paragraph's source doucment (maybe as a one file)
    if kwargs['dump_nd']:
        aug_docs = set()
        with open(kwargs['analysis_path'], 'r') as fp:
            aug_squad = json.load(fp)
            for aug_item in aug_squad:
                for doc_title in aug_item['eval_context_src']:
                    aug_docs.update([doc_title])
        print('Dump {} documents from neg pars'.format(len(aug_docs)))

        # Save tf-idf vectors of negative docs 
        title2idx = {}
        idx2title = {}
        doc_idxs = []
        aug_docs = sorted(list(aug_docs))
        for title_idx, title in enumerate(aug_docs):
            doc_idx = ranker.get_doc_index(title)
            doc_idxs.append(doc_idx)
            idx2title[doc_idx] = title
            title2idx[title] = doc_idx

        # Select doc idxs only
        doc_tfidf_mat = ranker.doc_mat[:,np.array(doc_idxs)].toarray()
        print(doc_tfidf_mat.shape, type(doc_tfidf_mat))
        doc_tfidf_mat = np.transpose(doc_tfidf_mat).tocsr()
        print(doc_tfidf_mat.shape, type(doc_tfidf_mat))
        exit()
        
        # Save as pickle
        doc_tfidf_path = os.path.join(context_tfidf_dir + 'neg_doc_mat.pkl')
        with open(doc_tfidf_path, 'wb') as f:
            pickle.dump(
                [title2idx, idx2title, doc_tfidf_mat], 
                f, protocol=pickle.HIGHEST_PROTOCOL
            )
        print('Negative Document TF-IDF saved as {}'.format(doc_tfidf_path))

    # Save tf-idf vector of documents
    if kwargs['dump_d']:
        title2spvec = {}
        for title, doc in tqdm(squad_docs.items()):
            doc_tfidf_emb = ranker.text2spvec(' '.join(doc))
            title2spvec[title] = doc_tfidf_emb

        # Save as pickle
        context_path = os.path.join(context_tfidf_dir, 'pos_doc_mat.pkl')
        with open(context_path, 'wb') as f:
            pickle.dump(title2spvec, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Document TF-IDF saved in {}'.format(context_tfidf_dir))

    # Save tf-idf vector of questions
    if kwargs['dump_q']:
        for q_id, que in tqdm(squad_ques.items()):
            que_tfidf_emb = ranker.text2spvec(que)
            question_path = os.path.join(
                question_tfidf_dir, q_id + '.tfidf'
            )
            save_npz(question_path, que_tfidf_emb)
        print('Question TF-IDF saved in {}'.format(question_tfidf_dir))


# Predefined paths
home = os.path.expanduser('~')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
SQUAD_PATH = os.path.join(home, 'data/squad', 'dev-v1.1.json')
ANALYSIS_PATH = os.path.join(home, 'data/squad',
    'dev-v1.1-large-rand-par100.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for dumping tf-idf')
    parser.add_argument('context_tfidf_dir', help='Context tfidf directory')
    parser.add_argument('question_tfidf_dir', help='Question tfidf directory')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--squad-path', type=str, default=SQUAD_PATH,
                        help='SQuAD dataset path')
    parser.add_argument('--analysis-path', type=str, default=ANALYSIS_PATH,
                        help='SQuAD dataset path')
    parser.add_argument('--dump-nd', default=False, action='store_true',
                        help='Dump negative docs from analysis file')
    parser.add_argument('--dump-d', default=False, action='store_true',
                        help='Dump docs from squad file')
    parser.add_argument('--dump-q', default=False, action='store_true',
                        help='Dump ques from squad file')
    args = parser.parse_args()

    dump_tfidf(**args.__dict__)
