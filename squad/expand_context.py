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
from expand_dev import _split_doc


# Predefined paths
home = os.path.expanduser('~')
DATA_PATH = os.path.join(home, 'data/squad/dev-v1.1.json')
RET_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data', 
    'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
DB_PATH = os.path.join(home, 'Desktop/Jinhyuk/github/DrQA/data',
    'wikipedia/docs.db')


if __name__ == '__main__':

    # Fixed arguments
    parser = argparse.ArgumentParser(description='Expand script for analysis')
    parser.add_argument('--data-path', type=str, default=DATA_PATH,
                        help='Dataset file path')
    parser.add_argument('--retriever-path', type=str, default=RET_PATH,
                        help='Document Retriever path')
    parser.add_argument('--db-path', type=str, default=DB_PATH,
                        help='Wikipedia DB path')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of process workers')
    parser.add_argument('--seed', type=int, default=999,
                        help='Random seed (for reproducibility)')

    # Controllable arguments
    parser.add_argument('--n-docs', type=int, default=30,
                        help='Number of closest documents per ex')
    parser.add_argument('--n-splits', type=int, default=100,
                        help='Number of json splits')
    args = parser.parse_args()

    print('# of closest docs: {}'.format(args.n_docs))

    # Prepare seed, dataset, DB
    random.seed(args.seed)
    np.random.seed(args.seed)
    with open(args.data_path, 'r') as fp:
        squad = json.load(fp)
    db = retriever.DocDB(db_path=args.db_path)

    # Make q2d or load
    if not os.path.exists('results/q2d_30.json'):
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
                    doc_len = [len(list(_split_doc(db.get_doc_text(doc)))[1:])
                               for doc in ranked[0]]
                    q2d[qas['id']] = [ranked[0], ranked[1].tolist(), doc_len]

        # Save as file
        with open('q2d_{}.json'.format(args.n_docs), 'w') as f:
            json.dump(q2d, f)
            exit()
    else:
        # Load saved file
        with open('results/q2d_30.json', 'r') as f:
            q2d = json.load(f)

    def space2udb(text):
        return '_'.join(text.split(' '))
    
    # Iterate, and append 
    new_squad = {'data': []}
    doc_titles = []
    overlap_docs = 0
    total_docs = 0
    for q, docs in tqdm(q2d.items()):
        total_docs += len(docs[0][:args.n_docs])
        for doc in docs[0][:args.n_docs]:
            if doc not in doc_titles:
                doc_titles.append(doc)
                text = list(_split_doc(db.get_doc_text(doc)))[1:]
                context_wrapper = [{'context': context} for context in text]
                article_wrapper = {
                    'title': space2udb(doc),
                    'paragraphs': context_wrapper
                }
                new_squad['data'].append(article_wrapper)
            else:
                overlap_docs += 1
    assert len(doc_titles) == len(new_squad['data'])

    print('# of total articles:', total_docs)
    print('# of overlap articles:', overlap_docs)
    print('# of final articles:', len(new_squad['data']))
    print('# of final paragraphs:', 
        sum([len(art['paragraphs']) for art in new_squad['data']]))

    # Sanity check
    for article in new_squad['data']:
        for para in article['paragraphs']:
            assert 'qas' not in para

    # Split and save
    split_size = len(new_squad['data']) // args.n_splits + 1
    for split_num in range(args.n_splits):
        file_name = 'dev-v1.1-top{}docs-{}.json'.format(args.n_docs, split_num)
        split_data = new_squad['data'][
            split_num*split_size:(split_num+1)*split_size
        ]
        with open(os.path.join('results/dev_contexts', file_name), 'w') as f:
            json.dump({'data': split_data}, f)
