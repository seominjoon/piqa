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

""" Identical to SQuAD v1.1 evaluation script"""
from collections import Counter
import string
import re
import argparse
import json
import sys


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
                print(c_emb.shape, q_emb.shape)
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


def get_predictions_c2q(context_emb_path, question_emb_path, c2q, threshold, sparse=False, metric='ip', progress=False):
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
    num_before = 0
    num_after = 0
    for cid, id_list in tqdm(c2q.items()):
        c_emb_path = os.path.join(context_emb_dir, '%s.npz' % cid)
        c_json_path = os.path.join(context_emb_dir, '%s.json' % cid)
        c_metadata_path = os.path.join(context_emb_dir, '%s.metadata' % cid)

        if not os.path.exists(c_emb_path):
            # print('Missing %s' % c_emb_path)
            continue
        if not os.path.exists(c_json_path):
            # print('Missing %s' % c_json_path)
            continue
        with open(c_metadata_path, 'r') as fp:
            metadata = json.load(fp)

        load = scipy.sparse.load_npz if sparse else np.load
        q_emb_mat = None
        for id_ in id_list:
            q_emb_path = os.path.join(question_emb_dir, '%s.npz' % id_)
            if not os.path.exists(q_emb_path):
                print('Missing %s' % q_emb_path)
            q_emb = load(q_emb_path)  # shape = [M, d], d is the embedding size.
            q_emb = q_emb['arr_0']
            q_emb_mat = np.append(q_emb_mat, q_emb, 0) \
                if q_emb_mat is not None else q_emb

        c_emb = load(c_emb_path)  # shape = [N, d], d is the embedding size.
        if not sparse:
            c_emb = c_emb['arr_0']

        with open(c_json_path, 'r') as fp:
            phrases = json.load(fp)

        booleans = np.array(metadata['probs']) >= threshold
        num_before += len(booleans)
        num_after += booleans.sum().item()
        c_emb = c_emb[booleans]
        phrases = [phrase for phrase, boolean in zip(phrases, booleans) if boolean]

        if sparse:
            raise Exception('Sparse not supported yet')
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

    ratio = float(num_before) / num_after

    return predictions, num_after, ratio


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


if __name__ == '__main__':
    squad_expected_version = '1.1'
    parser = argparse.ArgumentParser(description='Official merge script for PI-SQuAD v0.1')
    parser.add_argument('data_path', help='Dataset file path')
    parser.add_argument('context_emb_dir', help='Context embedding directory')
    parser.add_argument('question_emb_dir', help='Question embedding directory')
    parser.add_argument('output_path', help='Output path')
    parser.add_argument('--thresholds', default='0,.2,.4,.6,.8,.9,.95,.97,.99', type=str, help='probability thresholds')
    parser.add_argument('--sparse', default=False, action='store_true',
                        help='Whether the embeddings are scipy.sparse or pure numpy.')
    parser.add_argument('--metric', type=str, default='ip',
                        help='ip|l1|l2|cosine (inner product or L1 or L2 or cosine distance)')
    parser.add_argument('--progress', default=False, action='store_true', help='Show progress bar. Requires `tqdm`.')
    parser.add_argument('--q_mat', default=False, action='store_true', help='Query with matrix (faster)')
    args = parser.parse_args()

    thresholds = list(map(float, args.thresholds.split(',')))

    with open(args.data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != squad_expected_version:
            print('Evaluation expects v-' + squad_expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']

    if not args.q_mat:
        raise Exception()
        q2c = get_q2c(dataset)
        predictions = get_predictions(args.context_emb_dir,
                                      args.question_emb_dir, q2c, sparse=args.sparse,
                                      metric=args.metric, progress=args.progress)
    else:
        out = {'f1': [], 'em': [], 'num': [], 'ratio': [], 'thresholds': thresholds}
        c2q = get_c2q(dataset)
        for threshold in thresholds:
            predictions, num_after, ratio = get_predictions_c2q(args.context_emb_dir,
                                                                args.question_emb_dir, c2q, threshold,
                                                                sparse=args.sparse,
                                                                metric=args.metric, progress=args.progress)
            scores = evaluate(dataset, predictions)
            f1 = scores['f1']
            em = scores['exact_match']
            out['f1'].append(f1)
            out['em'].append(em)
            out['num'].append(num_after)
            out['ratio'].append(ratio)

        with open(args.output_path, 'w') as fp:
            json.dump(out, fp)
