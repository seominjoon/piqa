"""Official script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys

from scipy.sparse import save_npz
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

    # Get negative paragraph's source doucment
    aug_docs = set()
    with open(kwargs['analysis_path'], 'r') as fp:
        aug_squad = json.load(fp)
        for aug_item in aug_squad:
            for doc_title in aug_item['eval_context_src']:
                aug_docs.update([doc_title])
    print('Dump {} more documents from neg. paragraphs'.format(len(aug_docs)))

    # Save tf-idf vectors of negative docs (TODO: do not overlap with true docs)
    k = 0
    for title in tqdm(sorted(aug_docs)):
        doc_idx = ranker.get_doc_index(title)
        doc_tfidf_emb = ranker.doc_mat[doc_idx]
        title = title.replace('/', "_")
        context_path = os.path.join(
            context_tfidf_dir, '_'.join(title.split(' ')) + '.tfidf'
        )
        save_npz(context_path, doc_tfidf_emb)
    print('Negative Document TF-IDF saved in {}'.format(context_tfidf_dir))

    # Save tf-idf vector of documents
    for title, doc in tqdm(squad_docs.items()):
        doc_tfidf_emb = ranker.text2spvec(' '.join(doc))
        context_path = os.path.join(
            context_tfidf_dir, title + '.tfidf'
        )
        save_npz(context_path, doc_tfidf_emb)
    print('Document TF-IDF saved in {}'.format(context_tfidf_dir))

    # Save tf-idf vector of questions
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
    args = parser.parse_args()

    dump_tfidf(**args.__dict__)

