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


def get_q2c(dataset):
    q2c = {}
    for article in dataset:
        for para_idx, paragraph in enumerate(article['paragraphs']):
            cid = '%s_%d' % (article['title'], para_idx)
            for qa in paragraph['qas']:
                q2c[qa['id']] = cid
    return q2c


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
            elif metric == 'l1':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                m = scipy.sparse.linalg.norm(c_emb - q_emb, ord=2, axis=1)
        else:
            q_emb = q_emb['arr_0']
            c_emb = c_emb['arr_0']
            if metric == 'ip':
                sim = np.matmul(c_emb, q_emb.T)
                m = sim.max(1)
            elif metric == 'l1':
                m = numpy.linalg.norm(c_emb - q_emb, ord=1, axis=1)
            elif metric == 'l2':
                m = numpy.linalg.norm(c_emb - q_emb, ord=2, axis=1)

        argmax = m.argmax(0)
        predictions[id_] = phrases[argmax]
    
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
                        help='ip|l1|l2 (inner product or L1 or L2 distance)')
    parser.add_argument('--progress', default=False, action='store_true', help='Show progress bar. Requires `tqdm`.')
    args = parser.parse_args()

    with open(args.data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != squad_expected_version:
            print('Evaluation expects v-' + squad_expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    q2c = get_q2c(dataset)
    predictions = get_predictions(args.context_emb_dir, args.question_emb_dir, q2c, sparse=args.sparse,
                                  metric=args.metric, progress=args.progress)

    with open(args.pred_path, 'w') as fp:
        json.dump(predictions, fp)

