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


# For 1-to-1 q2c mapping
def get_q2c(dataset):
    q2c = {}
    for article in dataset:
        for para_idx, paragraph in enumerate(article['paragraphs']):
            cid = '%s_%d' % (article['title'], para_idx)
            for qa in paragraph['qas']:
                q2c[qa['id']] = cid
    return q2c


# For 1-to-many c2q mapping
def get_c2q(dataset):
    c2q = {}
    for article in dataset:
        for para_idx, paragraph in enumerate(article['paragraphs']):
            cid = '%s_%d' % (article['title'], para_idx)
            for qa in paragraph['qas']:
                if cid not in c2q:
                    c2q[cid] = []
                c2q[cid].append(qa['id'])
    return c2q


def get_predictions(context_emb_path, question_emb_path, q2c, sparse=False, metric='ip', progress=False):
    context_emb_dir, context_emb_ext = os.path.splitext(context_emb_path)
    question_emb_dir, question_emb_ext = os.path.splitext(question_emb_path)
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
    predictions = {}
    for id_, cid in tqdm(q2c.items()):
        q_emb_path = os.path.join(question_emb_dir, '%s.npz' % id_)
        c_emb_path = os.path.join(context_emb_dir, '%s.npz' % cid)
        c_json_path = os.path.join(context_emb_dir, '%s.json' % cid)

        if not os.path.exists(q_emb_path):
            print('Missing %s' % q_emb_path)
            continue
        if not os.path.exists(c_emb_path):
            print('Missing %s' % c_emb_path)
            continue
        if not os.path.exists(c_json_path):
            print('Missing %s' % c_json_path)
            continue

        load = scipy.sparse.load_npz if sparse else np.load
        q_emb = load(q_emb_path)  # shape = [M, d], d is the embedding size.
        c_emb = load(c_emb_path)  # shape = [N, d], d is the embedding size.

        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)

        if sparse:
            if metric == 'ip':
                sim = c_emb * q_emb.T
                m = sim.max(1)
                m = np.squeeze(np.array(m.todense()), 1)
            elif metric == 'cosine':
                c_emb = c_emb / scipy.sparse.linalg.norm(c_emb, ord=2, axis=1)
                q_emb = q_emb / scipy.sparse.linalg.norm(q_emb, ord=2, axis=1)
                sim = c_emb * q_emb.T
                m = sim.max(1)
                m = np.squeeze(np.array(m.todense()), 1)
            elif metric == 'l1':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=2, axis=1)
            else:
                raise ValueError(metric)
        else:
            q_emb = q_emb['arr_0']
            c_emb = c_emb['arr_0']
            if metric == 'ip':
                sim = np.matmul(c_emb, q_emb.T)
                # print(c_emb.shape, q_emb.shape)
                m = sim.max(1)
            elif metric == 'cosine':
                c_emb = c_emb / numpy.linalg.norm(c_emb, ord=2, axis=1, keepdims=True)
                q_emb = q_emb / numpy.linalg.norm(q_emb, ord=2, axis=0, keepdims=True)
                sim = np.matmul(c_emb, q_emb.T)
                m = sim.max(1)
            elif metric == 'l1':
                m = numpy.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                m = numpy.linalg.norm(c_emb - q_emb, ord=2, axis=1)
            else:
                raise ValueError(metric)

        argmax = m.argmax(0)
        predictions[id_] = phrases[argmax]
    
    if context_emb_ext == '.zip':
        shutil.rmtree(context_emb_dir)
    if question_emb_ext == '.zip':
        shutil.rmtree(question_emb_dir)

    return predictions


def get_predictions_c2q(context_emb_path, question_emb_path, c2q, sparse=False, metric='ip', progress=False, tfidf=False):
    context_emb_dir, context_emb_ext = os.path.splitext(context_emb_path)
    question_emb_dir, question_emb_ext = os.path.splitext(question_emb_path)
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
    predictions = {}
    for cid, id_list in tqdm(c2q.items()):
        if tfidf:
            c_emb_path = os.path.join(context_emb_dir, '%s_tfidf.npz' % cid)
        else:
            c_emb_path = os.path.join(context_emb_dir, '%s.npz' % cid)
        c_json_path = os.path.join(context_emb_dir, '%s.json' % cid)

        if not os.path.exists(c_emb_path):
            # print('Missing %s' % c_emb_path)
            continue
        if not os.path.exists(c_json_path):
            # print('Missing %s' % c_json_path)
            continue

        load = scipy.sparse.load_npz if (sparse or tfidf) else np.load
        q_emb_mat = None
        for id_ in id_list:
            if tfidf:
                q_emb_path = os.path.join(question_emb_dir, '%s_tfidf.npz' % id_)
            else:
                q_emb_path = os.path.join(question_emb_dir, '%s.npz' % id_)
            if not os.path.exists(q_emb_path):
                print('Missing %s' % q_emb_path)
            q_emb = load(q_emb_path)  # shape = [M, d], d is the embedding size.
            if not tfidf:
                q_emb = q_emb['arr_0']
                q_emb_mat = np.append(q_emb_mat, q_emb, 0) \
                            if q_emb_mat is not None else q_emb
            else:
                if q_emb_mat is None:
                    q_emb_mat = []
                q_emb_mat.append(q_emb)

        if tfidf:
            q_emb_mat = scipy.sparse.vstack(q_emb_mat)

        c_emb = load(c_emb_path)  # shape = [N, d], d is the embedding size.

        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)

        if sparse or tfidf:
            if metric == 'ip':
                sim = c_emb * q_emb_mat.T
                m = np.array(sim.todense())
                # m = sim.max(1)
                # m = np.squeeze(np.array(m.todense()), 1)
            elif metric == 'cosine':
                c_emb = c_emb / scipy.sparse.linalg.norm(c_emb, ord=2, axis=1)
                q_emb = q_emb / scipy.sparse.linalg.norm(q_emb, ord=2, axis=1)
                sim = c_emb * q_emb.T
                m = sim.max(1)
                m = np.squeeze(np.array(m.todense()), 1)
            elif metric == 'l1':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=2, axis=1)
            else:
                raise ValueError(metric)
        else:
            c_emb = c_emb['arr_0']
            if metric == 'ip':
                sim = np.matmul(c_emb, q_emb_mat.T)
                # print(c_emb.shape, q_emb_mat.shape)
                # m = sim.max(1) # Multiple query not allowed
                m = sim
            elif metric == 'cosine':
                raise NotImplementedError()
                c_emb = c_emb / numpy.linalg.norm(c_emb, ord=2, axis=1, keepdims=True)
                q_emb = q_emb / numpy.linalg.norm(q_emb, ord=2, axis=0, keepdims=True)
                sim = np.matmul(c_emb, q_emb.T)
                m = sim.max(1)
            elif metric == 'l1':
                raise NotImplementedError()
                m = numpy.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                raise NotImplementedError()
                m = numpy.linalg.norm(c_emb - q_emb, ord=2, axis=1)
            else:
                raise ValueError(metric)

        argmax = m.argmax(0)
        for idx, id_ in enumerate(id_list):
            predictions[id_] = phrases[argmax[idx]]
    
    if context_emb_ext == '.zip':
        shutil.rmtree(context_emb_dir)
    if question_emb_ext == '.zip':
        shutil.rmtree(question_emb_dir)

    return predictions


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='Official merge script for PI-SQuAD v0.1')
    parser.add_argument('data_path', help='Dataset file path')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('pred_path', help='Prediction json file path')
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='Whether the embeddings are scipy.sparse or pure numpy.')
    parser.add_argument('--metric', type=str, default='ip',
                        help='ip|l1|l2|cosine (inner product or L1 or L2 or cosine distance)')
    parser.add_argument('--progress', default=False, action='store_true', help='Show progress bar. Requires `tqdm`.')
    parser.add_argument('--q_mat', default=False, action='store_true', help='Query with matrix (faster)')
    parser.add_argument('--tfidf', default=False, action='store_true', help='use tfidf concatenated vectors')
    args = parser.parse_args()

    with open(args.data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != squad_expected_version:
            print('Evaluation expects v-' + squad_expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']

    if not args.q_mat:
        q2c = get_q2c(dataset)
        predictions = get_predictions(args.context_emb_dir, 
            args.question_emb_dir, q2c, sparse=args.sparse,
            metric=args.metric, progress=args.progress)
    else:
        c2q = get_c2q(dataset)
        predictions = get_predictions_c2q(args.context_emb_dir, 
            args.question_emb_dir, c2q, sparse=args.sparse,
            metric=args.metric, progress=args.progress, tfidf=args.tfidf)


    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)

