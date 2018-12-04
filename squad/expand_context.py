"""
Expand dev-1.1.json (or given) dataset using DrQA's Document Retriever

additional requirements:
    https://github.com/facebookresearch/DrQA
"""

import argparse
import regex
import os
import json
import numpy as np
import random

from tqdm import tqdm
from drqa import retriever, tokenizers
from drqa.retriever import utils


# Predefined paths
home = os.path.expanduser('~')
DATA_PATH = os.path.join(home, 'data/squad/dev-v1.1.json')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
DB_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data',
    'wikipedia/docs.db')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Expand script for analysis')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                        help='Dataset file path')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--db-path', type=str, default=DB_PATH,
                        help='Wikipedia DB path')
    parser.add_argument('--n-docs', type=int, default=30,
                        help='Number of closest documents per ex')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process workers')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed (for reproducibility)')
    parser.add_argument('--query-type', type=str, default='question',
                        help='context|question')
    parser.add_argument('--draft', default=False, action='store_true')
    args = parser.parse_args()

    print('# of closest docs: {}'.format(args.n_docs))
    print('Query type: {}'.format(args.query_type))

    # Prepare seed, dataset
    random.seed(args.seed)
    np.random.seed(args.seed)
    with open(args.data_path, 'r') as fp:
        squad = json.load(fp)

    # Make q2d or load
    if not os.path.exists('results/q2d_{}.json'.format(args.n_docs)):
        ranker = retriever.get_class('tfidf')(
            tfidf_path=args.retriever_path,
            strict=False
        )
        print('Retriever loaded from {}'.format(args.retriever_path))

        # Question to doc mapping
        q2d = {}
        for article in tqdm(squad['data']):
            for para in article['paragraphs']:
                for qas in para['qas']:
                    ranked = ranker.batch_closest_docs(
                        [qas['question']],
                        k=args.n_docs,
                        num_workers=args.num_workers
                    )[0]
                    q2d[qas['id']] = [ranked[0], ranked[1].tolist()]

        # Save as file
        with open('q2d_{}.json'.format(args.n_docs), 'w') as f:
            json.dump(q2d, f)
    else:
        # Load saved file
        with open('results/q2d_{}.json'.format(args.n_docs), 'r') as f:
            q2d = json.load(f)

    # Load Wikipedia DB
    db = retriever.DocDB(db_path=args.db_path)
    def udb2space(text):
        return ' '.join(text.split('_'))
    def space2udb(text):
        return '_'.join(text.split(' '))
    doc_titles = [udb2space(article['title']) for article in squad['data']]
    from expand_dev import _split_doc

    print('# of original articles:', len(squad['data']))
    
    # Iterate, and append 
    for q, docs in tqdm(q2d.items()):
        for doc in docs[0]:
            if doc not in doc_titles:
                doc_titles.append(doc)
                text = list(_split_doc(db.get_doc_text(doc)))[1:]
                context_wrapper = [{'context': context} for context in text]
                article_wrapper = {
                    'title': space2udb(doc),
                    'paragraphs': context_wrapper
                }
                squad['data'].append(article_wrapper)
    print('# of articles with augmentation:', len(squad['data']))

    # Remove qas
    for article in squad['data']:
        for para in article['paragraphs']:
            if 'qas' in para:
                del para['qas']

    # Save
    with open('dev-v1.1-top{}docs.json'.format(args.n_docs), 'w') as f:
        json.dump(squad, f)
