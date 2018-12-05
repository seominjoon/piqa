"""Official merge script for PI-SQuAD v0.1"""
from __future__ import print_function

import os
import argparse
import json
import sys
import pickle

import scipy.sparse
import numpy as np
import torch

from scipy.sparse import csr_matrix, hstack, vstack, save_npz, load_npz
from tqdm import tqdm


def merge_tfidf(p_emb_dir, q_emb_dir, q2d_path, tfidf_weight, sparse, 
                max_n_docs, cuda, **kwargs):

    # Load q2d mapping, scores, lengths
    with open(q2d_path, 'rb') as f:
        q2d = json.load(f)

    # qid||did => score
    qd_score = {}
    for qid, docs in q2d.items():
        for did, doc_score, _ in zip(*docs):
            qd_score[qid + '||' + did] = doc_score
    print('Number of questions to process: {}'.format(len(q2d)))
    print('Number of paragraphs to process: {}'.format(len(qd_score)))

    # Load/stack function for dense/sparse
    if not sparse:
        load = lambda x: np.load(x)['arr_0']
        stack = np.vstack
    else:
        load = lambda x: load_npz(x)
        stack = vstack

    predictions = {}
    for qid, docs in tqdm(q2d.items()):

        # Load question embedding vectors [1 X D]
        q_emb_path = os.path.join(q_emb_dir, qid + '.npz')
        assert os.path.exists(q_emb_path)
        q_emb = load(q_emb_path)

        # stores argmax of each document
        d_pred_phrases = []
        d_pred_scores = []

        # For each document, get best phrase with score 
        for n_docs, (did, dscore, dlen) in enumerate(zip(*docs)):

            # Load emb/json for each paragraph
            did_u = '_'.join(did.split(' '))
            pids = [did_u + '_{}'.format(k) for k in range(dlen)]
            p_emb_paths = [
                os.path.join(p_emb_dir, pid + '.npz') for pid in pids
                if os.path.exists(os.path.join(p_emb_dir, pid + '.npz'))
            ]
            p_json_paths = [
                os.path.join(p_emb_dir, pid + '.json') for pid in pids
                if os.path.exists(os.path.join(p_emb_dir, pid + '.json'))
            ]
            assert len(p_emb_paths) == len(p_json_paths)

            # TODO: Will be removed
            if len(p_emb_paths) < dlen:
                continue

            # Phrase embedding stack
            p_embs = [load(emb_path) for emb_path in p_emb_paths]
            p_embs = [emb for emb in p_embs if len(emb.shape) > 0]
            if len(p_embs) == 0:
                continue
            p_emb = stack(p_embs)
            
            # Phrase json
            phrases = []
            for json_path in p_json_paths:
                with open(json_path, 'r') as fp:
                    phrases += json.load(fp)
            assert len(phrases) == p_emb.shape[0]

            # TODO: could do answer aggregation here (strength-based)
            # Get top scored phrase from a single document
            scores = np.squeeze(p_emb.dot(q_emb.T), axis=1)
            if sparse:
                scores = scores.toarray()
            max_idx = np.argmax(scores)
            d_pred_phrases.append(phrases[max_idx])
            d_pred_scores.append(scores[max_idx] + dscore*tfidf_weight)

            if n_docs + 1 >= max_n_docs:
                break

        assert len(d_pred_phrases) == len(d_pred_scores)
        if len(d_pred_scores) > 0:
            mmax_idx = np.argmax(d_pred_scores) 
            predictions[qid] = d_pred_phrases[mmax_idx]
        else:
            predictions[qid] = ''

    return predictions


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='script for appending tf-idf')
    parser.add_argument('q2d_path', help='Dataset file path')
    parser.add_argument('p_emb_dir', help='Phrase embedding directory')
    parser.add_argument('q_emb_dir', help='Question embedding directory')
    parser.add_argument('pred_path', help='Prediction json file path')
    parser.add_argument('--max-n-docs', type=int, default=10)
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='If stored phrase vecs are sparse vec or not')
    parser.add_argument('--tfidf-weight', type=float, default=0,
                        help='TF-IDF vector weight')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='process np matrix with torch.cuda')
    args = parser.parse_args()

    # Merge using tfidf
    predictions = merge_tfidf(**args.__dict__)
    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)
