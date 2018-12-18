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


def merge_tfidf(qid2emb, p_emb_dir, d2q_path, context_path,
                tfidf_weight, sparse, 
                top_n_docs, cuda, **kwargs):

    # Load d2q mapping, and context file
    with open(context_path, 'rb') as f:
        contexts = json.load(f)
        doc_set = [d['title'] for d in contexts['data']]
    with open(d2q_path, 'rb') as f:
        d2q = json.load(f)
        new_d2q = {}
        for did, val in d2q.items():
            if '_'.join(did.split(' ')) in doc_set:
                new_d2q[did] = val
        d2q = new_d2q
        num_q = sum([len([v for v in q['qids'] if v[2] < top_n_docs]) 
                     for q in d2q.values()])
        uq = [[k[0] for k in [v for v in q['qids'] if v[2] < top_n_docs]] 
              for q in d2q.values()]
        uq = set([q for ql in uq for q in ql])

    print('Processing outputs of file: {}'.format(context_path))
    print('Number of documents to process: {}'.format(len(doc_set)))
    print('Number of questions to process: {}'.format(num_q))
    print('Number of unique questions to process: {}'.format(len(uq)))

    # Load/stack function for dense/sparse
    if not sparse:
        load = lambda x: np.load(x)['arr_0']
        stack = np.vstack
    else:
        load = lambda x: load_npz(x)
        stack = vstack

    num_q = 0
    predictions = {}
    for did, val in tqdm(d2q.items()):
        dlen = val['length']
        qlist = [v for v in val['qids'] if v[2] < top_n_docs]
        if len(qlist) == 0:
            continue
        num_q += len(qlist)

        # Load question embedding vectors [N X D]
        q_embs = []
        q_embs = [qid2emb[qid[0]] for qid in qlist]
        q_emb = stack(q_embs) if len(q_embs) > 1 else q_embs[0]

        # Load emb/json for each paragraph
        did_u = '_'.join(did.split(' ')).replace('/', '_')
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

        # Phrase embedding stack
        p_embs = [load(emb_path) for emb_path in p_emb_paths]
        p_embs = [emb for emb in p_embs if len(emb.shape) > 0]
        if len(p_embs) == 0:
            continue
        p_emb = stack(p_embs) if len(p_embs) > 1 else p_embs[0]
        
        # Phrase json
        phrases = []
        for json_path in p_json_paths:
            with open(json_path, 'r') as fp:
                phrases += json.load(fp)
        assert len(phrases) == p_emb.shape[0]

        # TODO: could do answer aggregation here (strength-based)
        # Get top scored phrase from a single document
        scores = q_emb.dot(p_emb.T)
        if sparse:
            scores = scores.toarray()
        max_idxs = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        assert len(max_scores) == len(qlist) == len(max_idxs)

        # Update prediction dict
        for max_score, max_idx, qitem in zip(max_scores, max_idxs, qlist):
            qid, qd_score, q_rank = qitem
            assert q_rank < top_n_docs
            if qid not in predictions:
                predictions[qid] = ['', -1e+9]
            new_cand = [phrases[max_idx], max_score + qd_score*tfidf_weight]
            predictions[qid] = predictions[qid] if predictions[qid][1] > \
                new_cand[1] else new_cand

    print('Number of questions processed: {}'.format(num_q))
    return predictions


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='script for appending tf-idf')
    parser.add_argument('p_emb_dir', help='Phrase embedding directory')
    parser.add_argument('q_emb_dir', help='Question embedding directory')
    parser.add_argument('d2q_path', help='Doc to que mapping file path')
    parser.add_argument('context_path', help='Context file directory')
    parser.add_argument('pred_path', help='Prediction json file path')
    parser.add_argument('--top-n-docs', type=int, default=10)
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='If stored phrase vecs are sparse vec or not')
    parser.add_argument('--tfidf-weight', type=float, default=0,
                        help='TF-IDF vector weight')
    parser.add_argument('--iteration', type=str, default='1')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='process np matrix with torch.cuda')
    args = parser.parse_args()

    # delete following lines for nsml-free implementation
    import nsml, shutil
    if nsml.IS_ON_NSML:
        if not os.path.exists(args.q_emb_dir):
            os.mkdir(args.q_emb_dir)
        if not os.path.exists(args.p_emb_dir):
            os.makedirs(args.p_emb_dir, exist_ok=True)

        # Load question vectors
        shutil.unpack_archive(
            os.path.join(
                nsml.DATASET_PATH,
                '{}_embed_dev-v1_1-question.zip'.format(
                    args.iteration,
                )
            ),
            args.q_emb_dir,
            format='zip',
        )
        print('q embed loaded in in', args.q_emb_dir)

        # Save them
        qid2emb = {}
        for qid in os.listdir(args.q_emb_dir):
            if os.path.isdir(os.path.join(args.q_emb_dir, qid)):
                print('Skipping directory: {}'.format(qid))
                continue
            qid_base = os.path.splitext(qid)[0]
            qid2emb[qid_base] = np.load(
                os.path.join(args.q_emb_dir, qid)
            )['arr_0']

        # Load phrase vectors
        shutil.unpack_archive(
            os.path.join(
                nsml.DATASET_PATH,
                '{}_embed_{}.zip'.format(
                    args.iteration,
                    os.path.splitext(os.path.basename(args.context_path))[0],
                ).replace('.', '_')
            ),
            args.p_emb_dir,
            format='zip',
        )
        print('p embeded in', filename)

        # Merge using tfidf
        predictions = merge_tfidf(qid2emb, **args.__dict__)
        with open(args.pred_path, 'w') as fp:
            json.dump(predictions, fp)

        # Remove files
        for filename in os.listdir(args.p_emb_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        for filename in os.listdir(args.q_emb_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    else:
        raise NotImplementedError
        # Merge using tfidf
        predictions = merge_tfidf(**args.__dict__)
        with open(args.pred_path, 'w') as fp:
            json.dump(predictions, fp)
